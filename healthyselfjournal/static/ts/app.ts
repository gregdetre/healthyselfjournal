/*
 * Minimal browser recorder for the Healthy Self Journal web interface.
 * Handles MediaRecorder capture, simple level metering, and uploads to the FastHTML server.
 */

type RecorderState = 'idle' | 'recording' | 'paused' | 'uploading';

type UploadResponse = UploadSuccessResponse | UploadErrorResponse;

interface RecorderConfig {
  uploadUrl: string;
  sessionId: string;
  shortDurationMs: number;
  shortVoicedMs: number;
  voiceRmsDbfsThreshold: number;
  voiceEnabled: boolean;
  ttsEndpoint?: string;
  ttsMime?: string;
  revealEndpoint?: string;
}

interface DOMElements {
  recordButton: HTMLButtonElement;
  statusText: HTMLElement;
  meterBar: HTMLElement;
  currentQuestion: HTMLElement;
  historyList: HTMLElement;
  totalDuration?: HTMLElement;
  recordTimer?: HTMLElement;
  recordTimerValue?: HTMLElement;
  voiceToggle?: HTMLInputElement;
  revealLink?: HTMLAnchorElement;
}

interface UploadSuccessResponse {
  status: 'ok';
  session_id: string;
  segment_label: string;
  duration_seconds: number;
  total_duration_seconds: number;
  total_duration_hms: string;
  total_duration_minutes_text?: string;
  transcript: string;
  next_question: string;
  llm_model?: string | null;
  summary_scheduled?: boolean;
  quit_after?: boolean;
}

interface UploadErrorResponse {
  status: 'error';
  error: string;
  detail?: string;
  http_status?: number;
}

class RecorderController {
  private state: RecorderState = 'idle';
  private mediaStream?: MediaStream;
  private mediaRecorder?: MediaRecorder;
  private audioContext?: AudioContext;
  private analyser?: AnalyserNode;
  private analyserBuffer?: Uint8Array<ArrayBuffer>;
  private sourceNode?: MediaStreamAudioSourceNode;
  private meterRafId?: number;
  private chunks: BlobPart[] = [];
  private startTimestamp = 0;
  private voicedMs = 0;
  private lastMeterSample = 0;
  private cancelFlag = false;
  private readonly rmsEpsilon = 1e-10; // avoid log10(0)
  private readonly voiceThresholdDbfs: number;
  private quitAfter = false;
  private audioEl?: HTMLAudioElement;
  private timerRafId?: number;
  private recordingStartMs = 0;
  private accumulatedMs = 0;

  constructor(private readonly elements: DOMElements, private readonly config: RecorderConfig) {
    this.voiceThresholdDbfs = config.voiceRmsDbfsThreshold ?? -40;
    this.elements.recordButton.addEventListener('click', () => {
      // If TTS is playing, stop it and immediately start recording
      if (this.audioEl && !this.audioEl.paused) {
        try {
          this.audioEl.pause();
          this.audioEl.currentTime = 0;
        } catch {}
      }
      void this.toggleRecording();
    });

    // Clickable session ID to reveal the markdown file in Finder (macOS)
    if (this.elements.revealLink && this.config.revealEndpoint) {
      this.elements.revealLink.addEventListener('click', (ev: MouseEvent) => {
        ev.preventDefault();
        void fetch(this.config.revealEndpoint!, { method: 'POST' }).catch(() => {});
      });
    }

    window.addEventListener('keydown', (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        this.cancelRecording();
        return;
      }
      if ((event.key === 'q' || event.key === 'Q') && (this.state === 'recording' || this.state === 'paused')) {
        event.preventDefault();
        this.quitAfter = true;
        this.stopRecording();
        return;
      }
      if (event.code === 'Space' || event.key === ' ') {
        // SPACE: pause/resume during recording
        event.preventDefault();
        this.togglePause();
        return;
      }
      if (event.key === 'Enter') {
        event.preventDefault();
        // If TTS is playing, stop it and immediately start recording
        if (this.audioEl && !this.audioEl.paused) {
          try {
            this.audioEl.pause();
            this.audioEl.currentTime = 0;
          } catch {}
        }
        void this.toggleRecording();
      }
    });

    if (!navigator.mediaDevices?.getUserMedia) {
      this.fail('Microphone access is not supported in this browser.');
    }

    if (typeof MediaRecorder === 'undefined') {
      this.fail('MediaRecorder is unavailable. Please use a modern Chromium-based browser.');
    }

    // Optional voice mode toggle (runtime on/off)
    if (this.elements.voiceToggle) {
      this.elements.voiceToggle.checked = !!this.config.voiceEnabled;
      this.elements.voiceToggle.addEventListener('change', () => {
        const enabled = !!this.elements.voiceToggle?.checked;
        this.config.voiceEnabled = enabled;
        if (!enabled && this.audioEl && !this.audioEl.paused) {
          try {
            this.audioEl.pause();
            this.audioEl.currentTime = 0;
          } catch {}
          // If we had been waiting for TTS to finish before recording, proceed now
          if (this.state === 'idle') {
            this.autoStartAfterQuestion();
          }
        }
      });
    }

    // Optionally speak the initial question, then auto-start recording
    // In voice mode, wait until playback has started; recording begins after TTS ends
    const initialQuestion = this.elements.currentQuestion.textContent ?? '';
    if (this.config.voiceEnabled && this.config.ttsEndpoint) {
      void this.playTts(initialQuestion).finally(() => {
        this.autoStartAfterQuestion();
      });
    } else {
      // Non-voice mode: start immediately after question is displayed
      setTimeout(() => {
        this.autoStartAfterQuestion();
      }, 0);
    }
  }

  private fail(message: string): void {
    this.setStatus(message);
    this.elements.recordButton.disabled = true;
  }

  private async toggleRecording(): Promise<void> {
    if (this.state === 'uploading') {
      return;
    }
    if (this.state === 'idle') {
      await this.startRecording();
      return;
    }
    if (this.state === 'recording' || this.state === 'paused') {
      this.stopRecording();
    }
  }

  private async startRecording(): Promise<void> {
    try {
      this.elements.recordButton.disabled = true;
      this.quitAfter = false;
      if (!this.mediaStream) {
        this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      }
      if (!this.audioContext) {
        this.audioContext = new AudioContext();
      }
      await this.audioContext.resume();

      this.sourceNode = this.audioContext.createMediaStreamSource(this.mediaStream);
      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 2048;
      this.sourceNode.connect(this.analyser);
      this.analyserBuffer = new Uint8Array(new ArrayBuffer(this.analyser.fftSize));

      const mimeType = this.chooseMimeType();
      this.mediaRecorder = new MediaRecorder(this.mediaStream, { mimeType });
      this.mediaRecorder.addEventListener('dataavailable', this.handleDataAvailable);
      this.mediaRecorder.addEventListener('stop', this.handleStop);

      this.chunks = [];
      this.voicedMs = 0;
      this.lastMeterSample = performance.now();
      this.startTimestamp = performance.now();
      this.recordingStartMs = performance.now();
      this.accumulatedMs = 0;
      this.mediaRecorder.start();

      this.state = 'recording';
      this.elements.recordButton.disabled = false;
      this.elements.recordButton.textContent = 'Stop recording';
      this.elements.recordButton.dataset.state = 'recording';
      this.setStatus('Recording… Press SPACE to pause, ENTER to stop.');
      this.startMeterLoop();
      this.startTimerLoop();
    } catch (error) {
      console.error('Failed to start recording', error);
      this.setStatus('Microphone permission denied or recorder unavailable.');
      this.resetButton();
    }
  }

  private stopRecording(): void {
    if ((this.state !== 'recording' && this.state !== 'paused') || !this.mediaRecorder) {
      return;
    }
    this.state = 'uploading';
    this.cancelFlag = false;
    this.elements.recordButton.disabled = true;
    this.elements.recordButton.textContent = 'Processing…';
    this.elements.recordButton.dataset.state = 'uploading';
    this.setStatus('Processing audio…');
    this.stopMeterLoop();
    this.stopTimerLoop(true);
    this.mediaRecorder.stop();
  }

  private cancelRecording(): void {
    if ((this.state !== 'recording' && this.state !== 'paused') || !this.mediaRecorder) {
      return;
    }
    this.cancelFlag = true;
    this.quitAfter = false;
    this.state = 'idle';
    this.stopMeterLoop();
    this.stopTimerLoop(true);
    this.mediaRecorder.stop();
  }

  private togglePause(): void {
    if (!this.mediaRecorder) return;
    if (this.state === 'recording') {
      try {
        if (this.mediaRecorder.state === 'recording') {
          this.mediaRecorder.pause();
        }
      } catch {}
      this.state = 'paused';
      this.stopMeterLoop();
      // Accumulate elapsed and pause timer updates
      if (this.recordingStartMs > 0) {
        this.accumulatedMs += Math.max(performance.now() - this.recordingStartMs, 0);
        this.recordingStartMs = 0;
      }
      this.stopTimerLoop(false);
      this.elements.recordButton.textContent = 'Resume recording';
      this.elements.recordButton.dataset.state = 'paused';
      this.setStatus('Paused. Press SPACE to resume, ENTER to stop.');
      return;
    }
    if (this.state === 'paused') {
      try {
        if (this.mediaRecorder.state === 'paused') {
          this.mediaRecorder.resume();
        }
      } catch {}
      this.state = 'recording';
      this.elements.recordButton.textContent = 'Stop recording';
      this.elements.recordButton.dataset.state = 'recording';
      this.setStatus('Recording… Press SPACE to pause, ENTER to stop.');
      this.startMeterLoop();
      // Resume timer updates
      this.recordingStartMs = performance.now();
      this.startTimerLoop();
      return;
    }
  }

  private handleDataAvailable = (event: BlobEvent): void => {
    if (event.data && event.data.size > 0) {
      this.chunks.push(event.data);
    }
  };

  private handleStop = async (): Promise<void> => {
    const blob = new Blob(this.chunks, {
      type: this.mediaRecorder?.mimeType || 'audio/webm;codecs=opus',
    });
    this.chunks = [];

    if (this.mediaRecorder) {
      this.mediaRecorder.removeEventListener('dataavailable', this.handleDataAvailable);
      this.mediaRecorder.removeEventListener('stop', this.handleStop);
      this.mediaRecorder = undefined;
    }

    if (this.sourceNode) {
      this.sourceNode.disconnect();
      this.sourceNode = undefined;
    }

    const durationMs = Math.max(this.getCurrentElapsedMs(), 0);
    const voicedMs = this.voicedMs;
    this.voicedMs = 0;
    this.lastMeterSample = 0;

    if (this.cancelFlag) {
      this.cancelFlag = false;
      this.quitAfter = false;
      this.resetButton();
      this.setStatus('Recording cancelled.');
      this.state = 'idle';
      this.accumulatedMs = 0;
      this.recordingStartMs = 0;
      this.stopTimerLoop(true);
      return;
    }

    if (durationMs <= this.config.shortDurationMs && voicedMs <= this.config.shortVoicedMs) {
      this.quitAfter = false;
      this.resetButton();
      const dSec = (durationMs / 1000).toFixed(2);
      const vSec = (voicedMs / 1000).toFixed(2);
      const thDur = (this.config.shortDurationMs / 1000).toFixed(2);
      const thVoiced = (this.config.shortVoicedMs / 1000).toFixed(2);
      this.setStatus(
        `Discarded (short/quiet): duration ${dSec}s (≤ ${thDur}s) & voiced ${vSec}s (≤ ${thVoiced}s). Try again.`,
      );
      this.state = 'idle';
      return;
    }

    this.setStatus('Uploading clip…');
    this.elements.recordButton.textContent = 'Uploading…';
    this.elements.recordButton.disabled = true;

    try {
      const response = await this.uploadClip(blob, durationMs, voicedMs, this.quitAfter);
      if (response.status !== 'ok') {
        this.handleUploadError(response);
        return;
      }
      this.setStatus('Thinking about the next question…');
      this.handleUploadSuccess(response, durationMs);
      this.setStatus('Ready for the next reflection.');
    } catch (error) {
      console.error('Upload failed', error);
      this.setStatus('Upload failed. Please check your connection and try again.');
    } finally {
      this.quitAfter = false;
      this.resetButton();
      this.state = 'idle';
      this.accumulatedMs = 0;
      this.recordingStartMs = 0;
    }
  };

  private startMeterLoop(): void {
    if (!this.analyser || !this.analyserBuffer) {
      return;
    }

    const update = (): void => {
      if (!this.analyser || !this.analyserBuffer) {
        return;
      }
      this.analyser.getByteTimeDomainData(this.analyserBuffer);
      let sumSquares = 0;
      for (let i = 0; i < this.analyserBuffer.length; i += 1) {
        const value = this.analyserBuffer[i] - 128;
        sumSquares += value * value;
      }
      const rms = Math.sqrt(sumSquares / this.analyserBuffer.length) / 128;
      const level = Math.min(1, rms * 2);
      this.setMeterLevel(level);

      const now = performance.now();
      const delta = Math.max(now - this.lastMeterSample, 0);
      // Convert amplitude RMS (0..1) to dBFS and compare with configured threshold
      const dbfs = 20 * Math.log10(Math.max(rms, this.rmsEpsilon));
      if (dbfs >= this.voiceThresholdDbfs) {
        this.voicedMs += delta;
      }
      this.lastMeterSample = now;

      if (this.state === 'recording') {
        this.meterRafId = requestAnimationFrame(update);
      } else {
        this.setMeterLevel(0);
      }
    };

    this.lastMeterSample = performance.now();
    this.meterRafId = requestAnimationFrame(update);
  }

  private stopMeterLoop(): void {
    if (this.meterRafId !== undefined) {
      cancelAnimationFrame(this.meterRafId);
      this.meterRafId = undefined;
    }
    this.setMeterLevel(0);
  }

  private async uploadClip(
    blob: Blob,
    durationMs: number,
    voicedMs: number,
    quitAfter: boolean,
  ): Promise<UploadResponse> {
    const form = new FormData();
    const filename = `browser-${Date.now()}.webm`;
    form.append('audio', blob, filename);
    form.append('mime', blob.type || 'audio/webm');
    form.append('duration_ms', durationMs.toString());
    form.append('voiced_ms', voicedMs.toString());
    form.append('quit_after', quitAfter ? '1' : '0');

    const response = await fetch(this.config.uploadUrl, {
      method: 'POST',
      body: form,
    });

    const payload = (await response.json()) as UploadResponse;
    if (!response.ok && payload.status === 'error') {
      payload.http_status = response.status;
    }
    return payload;
  }

  private handleUploadSuccess(payload: UploadSuccessResponse, durationMs: number): void {
    const answeredQuestion = this.elements.currentQuestion.textContent ?? '';
    this.elements.currentQuestion.textContent = payload.next_question;
    const shouldQuitAfter = !!payload.quit_after;

    const historyItem = document.createElement('article');
    historyItem.className = 'hsj-exchange';

    const qHeading = document.createElement('h3');
    qHeading.textContent = 'AI';
    const qBody = document.createElement('p');
    qBody.textContent = answeredQuestion;

    const aHeading = document.createElement('h3');
    aHeading.textContent = 'You';
    const aBody = document.createElement('p');
    aBody.textContent = payload.transcript;

    const meta = document.createElement('p');
    meta.className = 'hsj-exchange-meta';
    meta.textContent = `Segment ${payload.segment_label} · ${(durationMs / 1000).toFixed(1)}s`;

    historyItem.append(qHeading, qBody, aHeading, aBody, meta);
    this.elements.historyList.prepend(historyItem);

    // Update cumulative duration display (minutes-only text when available)
    if (this.elements.totalDuration) {
      this.elements.totalDuration.textContent =
        payload.total_duration_minutes_text || payload.total_duration_hms;
    }

    if (shouldQuitAfter) {
      this.setStatus('Session complete (quit-after).');
      return;
    }

    // Optionally speak the next question using server-side TTS, then auto-start recording
    if (this.config.voiceEnabled && this.config.ttsEndpoint) {
      void this.playTts(payload.next_question).finally(() => {
        this.autoStartAfterQuestion();
      });
    } else {
      this.autoStartAfterQuestion();
    }
  }

  private handleUploadError(payload: UploadErrorResponse): void {
    this.quitAfter = false;
    const detail = payload.detail ? ` ${payload.detail}` : '';
    switch (payload.error) {
      case 'short_answer_discarded':
        this.setStatus(
          'Server discarded a very short or quiet clip. Try again when ready.',
        );
        break;
      case 'audio_format_unsupported':
        this.setStatus(
          'Upload rejected: speech-to-text backend cannot process this audio format.' + detail,
        );
        break;
      case 'upload_too_large':
        this.setStatus('Upload exceeded the server size limit. Try a shorter reflection.');
        break;
      case 'inactive_session':
      case 'unknown_session':
        this.setStatus('Session is no longer active. Refresh the page to resume.');
        break;
      case 'processing_failed':
        this.setStatus('Processing failed on the server. Check logs and try again.' + detail);
        break;
      case 'question_failed':
        this.setStatus(
          'Answer saved but the next question could not be generated. Check logs.' + detail,
        );
        break;
      default: {
        const hint = payload.error ? ` (${payload.error})` : '';
        this.setStatus(`Upload failed${hint}.${detail}`.trim());
        break;
      }
    }
  }

  private async playTts(text: string): Promise<void> {
    try {
      if (!text.trim()) return;
      if (!this.config.voiceEnabled) return;
      const endpoint = this.config.ttsEndpoint!;
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (!response.ok) {
        this.setStatus('TTS unavailable; continuing silently.');
        return; // Fail soft; keep UX responsive
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      if (!this.audioEl) {
        this.audioEl = new Audio();
      }
      this.audioEl.src = url;
      if (this.config.voiceEnabled) {
        await this.audioEl.play().catch(() => {});
      }
      // Revoke after some delay to ensure playback is not interrupted
      setTimeout(() => URL.revokeObjectURL(url), 10_000);
    } catch (err) {
      this.setStatus('TTS unavailable; continuing silently.');
    }
  }

  private autoStartAfterQuestion(): void {
    // Do not interrupt ongoing upload or recording
    if (this.state !== 'idle') return;

    // If TTS is enabled and currently playing, wait for it to finish
    if (this.config.voiceEnabled && this.audioEl && !this.audioEl.paused) {
      const onEnded = (): void => {
        this.audioEl?.removeEventListener('ended', onEnded);
        if (this.state === 'idle') {
          void this.toggleRecording();
        }
      };
      this.audioEl.addEventListener('ended', onEnded, { once: true });
      return;
    }

    void this.toggleRecording();
  }

  private chooseMimeType(): string {
    const preferred = 'audio/webm;codecs=opus';
    if (MediaRecorder.isTypeSupported?.(preferred)) {
      return preferred;
    }
    return 'audio/webm';
  }

  private setMeterLevel(level: number): void {
    this.elements.meterBar.style.setProperty('--meter-level', level.toString());
  }

  private setStatus(message: string): void {
    this.elements.statusText.textContent = message;
  }

  private resetButton(): void {
    this.elements.recordButton.disabled = false;
    this.elements.recordButton.textContent = 'Start recording';
    this.elements.recordButton.dataset.state = 'idle';
  }

  private startTimerLoop(): void {
    const container = this.elements.recordTimer;
    const valueEl = this.elements.recordTimerValue;
    if (!container || !valueEl) return;
    container.hidden = false;

    const update = (): void => {
      const elapsed = this.getCurrentElapsedMs();
      valueEl.textContent = this.formatElapsed(elapsed);
      if (this.state === 'recording') {
        this.timerRafId = requestAnimationFrame(update);
      }
    };

    // Immediate update, then animate
    valueEl.textContent = this.formatElapsed(this.getCurrentElapsedMs());
    this.timerRafId = requestAnimationFrame(update);
  }

  private stopTimerLoop(hide: boolean): void {
    if (this.timerRafId !== undefined) {
      cancelAnimationFrame(this.timerRafId);
      this.timerRafId = undefined;
    }
    if (hide && this.elements.recordTimer) {
      this.elements.recordTimer.hidden = true;
    }
  }

  private getCurrentElapsedMs(): number {
    const running = this.state === 'recording' && this.recordingStartMs > 0;
    const current = running ? Math.max(performance.now() - this.recordingStartMs, 0) : 0;
    return this.accumulatedMs + current;
  }

  private formatElapsed(ms: number): string {
    const totalSeconds = Math.floor(ms / 1000);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    if (hours > 0) {
      return `${hours}:${minutes.toString().padStart(2, '0')}:${seconds
        .toString()
        .padStart(2, '0')}`;
    }
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }
}

function bootstrap(): void {
  const body = document.body;
  const uploadUrl = body.dataset.uploadUrl;
  const sessionId = body.dataset.sessionId;
  if (!uploadUrl || !sessionId) {
    console.error('Missing session metadata on <body>.');
    return;
  }

  const shortDurationSeconds = Number(body.dataset.shortDuration ?? '1.2');
  const shortVoicedSeconds = Number(body.dataset.shortVoiced ?? '0.6');
  const voiceRmsDbfsThreshold = Number(body.dataset.voiceRmsDbfsThreshold ?? '-40');
  const voiceEnabled = (body.dataset.voiceEnabled ?? 'false') === 'true';
  const ttsEndpoint = body.dataset.ttsEndpoint;
  const ttsMime = body.dataset.ttsMime;
  const revealEndpoint = body.dataset.revealEndpoint;

  const elements: DOMElements = {
    recordButton: document.getElementById('record-button') as HTMLButtonElement,
    statusText: document.getElementById('status-text') as HTMLElement,
    meterBar: document.querySelector('#level-meter .bar') as HTMLElement,
    currentQuestion: document.getElementById('current-question') as HTMLElement,
    historyList: document.getElementById('history-list') as HTMLElement,
    totalDuration: document.getElementById('total-duration') as HTMLElement | null || undefined,
    recordTimer: document.getElementById('record-timer') as HTMLElement | null || undefined,
    recordTimerValue: document.getElementById('record-timer-value') as HTMLElement | null || undefined,
    voiceToggle: document.getElementById('voice-toggle') as HTMLInputElement | null || undefined,
    revealLink: document.getElementById('reveal-session') as HTMLAnchorElement | null || undefined,
  };

  // Ensure initial total duration text reflects minutes-only value if provided on body dataset
  if (elements.totalDuration) {
    const minutesText = body.dataset.totalMinutesText;
    if (minutesText) {
      elements.totalDuration.textContent = minutesText;
    }
  }

  if (!elements.recordButton || !elements.statusText || !elements.meterBar) {
    console.error('Missing critical DOM elements.');
    return;
  }

  const config: RecorderConfig = {
    uploadUrl,
    sessionId,
    shortDurationMs: shortDurationSeconds * 1000,
    shortVoicedMs: shortVoicedSeconds * 1000,
    voiceRmsDbfsThreshold,
    // If voice toggle is disabled in DOM (privacy), force voice off client-side
    voiceEnabled: voiceEnabled && !(document.getElementById('voice-toggle') as HTMLInputElement | null)?.disabled,
    ttsEndpoint,
    ttsMime,
    revealEndpoint,
  };

  new RecorderController(elements, config);

  // Wire global Reveal Sessions Folder link to POST to avoid navigation
  const revealSessionsLink = document.getElementById('reveal-sessions-folder') as HTMLAnchorElement | null;
  if (revealSessionsLink) {
    revealSessionsLink.addEventListener('click', (ev) => {
      ev.preventDefault();
      fetch('/reveal/sessions', { method: 'POST' }).catch(() => {});
    });
  }
}

document.addEventListener('DOMContentLoaded', bootstrap);
