"use strict";
/*
 * Minimal browser recorder for the Healthy Self Journal web interface.
 * Handles MediaRecorder capture, simple level metering, and uploads to the FastHTML server.
 */
class RecorderController {
    constructor(elements, config) {
        this.elements = elements;
        this.config = config;
        this.state = 'idle';
        this.chunks = [];
        this.startTimestamp = 0;
        this.voicedMs = 0;
        this.lastMeterSample = 0;
        this.cancelFlag = false;
        this.voicedThreshold = 0.07; // heuristic amplitude threshold 0-1
        this.recordingStartMs = 0;
        this.accumulatedMs = 0;
        this.handleDataAvailable = (event) => {
            if (event.data && event.data.size > 0) {
                this.chunks.push(event.data);
            }
        };
        this.handleStop = async () => {
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
                this.resetButton();
                this.setStatus('Recording cancelled.');
                this.state = 'idle';
                this.accumulatedMs = 0;
                this.recordingStartMs = 0;
                this.stopTimerLoop(true);
                return;
            }
            if (durationMs < this.config.shortDurationMs || voicedMs < this.config.shortVoicedMs) {
                this.resetButton();
                this.setStatus('Discarded a very short or quiet clip. Try again when ready.');
                this.state = 'idle';
                return;
            }
            this.setStatus('Uploading clip…');
            this.elements.recordButton.textContent = 'Uploading…';
            this.elements.recordButton.disabled = true;
            try {
                const response = await this.uploadClip(blob, durationMs, voicedMs);
                if (response.status !== 'ok') {
                    throw new Error(response.error || 'Unknown upload error');
                }
                this.setStatus('Thinking about the next question…');
                this.handleUploadSuccess(response, durationMs);
                this.setStatus('Ready for the next reflection.');
            }
            catch (error) {
                console.error('Upload failed', error);
                this.setStatus('Upload failed. Please check your connection and try again.');
            }
            finally {
                this.resetButton();
                this.state = 'idle';
                this.accumulatedMs = 0;
                this.recordingStartMs = 0;
            }
        };
        this.elements.recordButton.addEventListener('click', () => {
            // If TTS is playing, stop it and immediately start recording
            if (this.audioEl && !this.audioEl.paused) {
                try {
                    this.audioEl.pause();
                    this.audioEl.currentTime = 0;
                }
                catch { }
            }
            void this.toggleRecording();
        });
        window.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                this.cancelRecording();
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
                    }
                    catch { }
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
                    }
                    catch { }
                    // If we had been waiting for TTS to finish before recording, proceed now
                    if (this.state === 'idle') {
                        this.autoStartAfterQuestion();
                    }
                }
            });
        }
        // Optionally speak the initial question, then auto-start recording
        if (this.config.voiceEnabled && this.config.ttsEndpoint) {
            void this.playTts(this.elements.currentQuestion.textContent ?? '');
        }
        // Auto-start recording after the opening question is displayed
        // Mirrors CLI behaviour of starting immediately
        setTimeout(() => {
            this.autoStartAfterQuestion();
        }, 0);
    }
    fail(message) {
        this.setStatus(message);
        this.elements.recordButton.disabled = true;
    }
    async toggleRecording() {
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
    async startRecording() {
        try {
            this.elements.recordButton.disabled = true;
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
        }
        catch (error) {
            console.error('Failed to start recording', error);
            this.setStatus('Microphone permission denied or recorder unavailable.');
            this.resetButton();
        }
    }
    stopRecording() {
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
    cancelRecording() {
        if ((this.state !== 'recording' && this.state !== 'paused') || !this.mediaRecorder) {
            return;
        }
        this.cancelFlag = true;
        this.state = 'idle';
        this.stopMeterLoop();
        this.stopTimerLoop(true);
        this.mediaRecorder.stop();
    }
    togglePause() {
        if (!this.mediaRecorder)
            return;
        if (this.state === 'recording') {
            try {
                if (this.mediaRecorder.state === 'recording') {
                    this.mediaRecorder.pause();
                }
            }
            catch { }
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
            }
            catch { }
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
    startMeterLoop() {
        if (!this.analyser || !this.analyserBuffer) {
            return;
        }
        const update = () => {
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
            if (rms > this.voicedThreshold) {
                this.voicedMs += delta;
            }
            this.lastMeterSample = now;
            if (this.state === 'recording') {
                this.meterRafId = requestAnimationFrame(update);
            }
            else {
                this.setMeterLevel(0);
            }
        };
        this.lastMeterSample = performance.now();
        this.meterRafId = requestAnimationFrame(update);
    }
    stopMeterLoop() {
        if (this.meterRafId !== undefined) {
            cancelAnimationFrame(this.meterRafId);
            this.meterRafId = undefined;
        }
        this.setMeterLevel(0);
    }
    async uploadClip(blob, durationMs, voicedMs) {
        const form = new FormData();
        const filename = `browser-${Date.now()}.webm`;
        form.append('audio', blob, filename);
        form.append('mime', blob.type || 'audio/webm');
        form.append('duration_ms', durationMs.toString());
        form.append('voiced_ms', voicedMs.toString());
        form.append('question', this.elements.currentQuestion.textContent ?? '');
        const response = await fetch(this.config.uploadUrl, {
            method: 'POST',
            body: form,
        });
        const payload = (await response.json());
        if (!response.ok) {
            throw new Error(payload.status === 'error'
                ? payload.error
                : `Upload failed with status ${response.status}`);
        }
        return payload;
    }
    handleUploadSuccess(payload, durationMs) {
        const answeredQuestion = this.elements.currentQuestion.textContent ?? '';
        this.elements.currentQuestion.textContent = payload.next_question;
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
        // Update cumulative duration display
        if (this.elements.totalDuration) {
            this.elements.totalDuration.textContent = payload.total_duration_hms;
        }
        // Optionally speak the next question using server-side TTS, then auto-start recording
        if (this.config.voiceEnabled && this.config.ttsEndpoint) {
            void this.playTts(payload.next_question).finally(() => {
                this.autoStartAfterQuestion();
            });
        }
        else {
            this.autoStartAfterQuestion();
        }
    }
    async playTts(text) {
        try {
            if (!text.trim())
                return;
            if (!this.config.voiceEnabled)
                return;
            const endpoint = this.config.ttsEndpoint;
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text }),
            });
            if (!response.ok) {
                return; // Fail soft; keep UX responsive
            }
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            if (!this.audioEl) {
                this.audioEl = new Audio();
            }
            this.audioEl.src = url;
            if (this.config.voiceEnabled) {
                await this.audioEl.play().catch(() => { });
            }
            // Revoke after some delay to ensure playback is not interrupted
            setTimeout(() => URL.revokeObjectURL(url), 10000);
        }
        catch (err) {
            // Silently ignore TTS errors for now
        }
    }
    autoStartAfterQuestion() {
        // Do not interrupt ongoing upload or recording
        if (this.state !== 'idle')
            return;
        // If TTS is enabled and currently playing, wait for it to finish
        if (this.config.voiceEnabled && this.audioEl && !this.audioEl.paused) {
            const onEnded = () => {
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
    chooseMimeType() {
        const preferred = 'audio/webm;codecs=opus';
        if (MediaRecorder.isTypeSupported?.(preferred)) {
            return preferred;
        }
        return 'audio/webm';
    }
    setMeterLevel(level) {
        this.elements.meterBar.style.setProperty('--meter-level', level.toString());
    }
    setStatus(message) {
        this.elements.statusText.textContent = message;
    }
    resetButton() {
        this.elements.recordButton.disabled = false;
        this.elements.recordButton.textContent = 'Start recording';
        this.elements.recordButton.dataset.state = 'idle';
    }
    startTimerLoop() {
        const container = this.elements.recordTimer;
        const valueEl = this.elements.recordTimerValue;
        if (!container || !valueEl)
            return;
        container.hidden = false;
        const update = () => {
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
    stopTimerLoop(hide) {
        if (this.timerRafId !== undefined) {
            cancelAnimationFrame(this.timerRafId);
            this.timerRafId = undefined;
        }
        if (hide && this.elements.recordTimer) {
            this.elements.recordTimer.hidden = true;
        }
    }
    getCurrentElapsedMs() {
        const running = this.state === 'recording' && this.recordingStartMs > 0;
        const current = running ? Math.max(performance.now() - this.recordingStartMs, 0) : 0;
        return this.accumulatedMs + current;
    }
    formatElapsed(ms) {
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
function bootstrap() {
    const body = document.body;
    const uploadUrl = body.dataset.uploadUrl;
    const sessionId = body.dataset.sessionId;
    if (!uploadUrl || !sessionId) {
        console.error('Missing session metadata on <body>.');
        return;
    }
    const shortDurationSeconds = Number(body.dataset.shortDuration ?? '1.2');
    const shortVoicedSeconds = Number(body.dataset.shortVoiced ?? '0.6');
    const voiceEnabled = (body.dataset.voiceEnabled ?? 'false') === 'true';
    const ttsEndpoint = body.dataset.ttsEndpoint;
    const ttsMime = body.dataset.ttsMime;
    const elements = {
        recordButton: document.getElementById('record-button'),
        statusText: document.getElementById('status-text'),
        meterBar: document.querySelector('#level-meter .bar'),
        currentQuestion: document.getElementById('current-question'),
        historyList: document.getElementById('history-list'),
        totalDuration: document.getElementById('total-duration') || undefined,
        recordTimer: document.getElementById('record-timer') || undefined,
        recordTimerValue: document.getElementById('record-timer-value') || undefined,
        voiceToggle: document.getElementById('voice-toggle') || undefined,
    };
    if (!elements.recordButton || !elements.statusText || !elements.meterBar) {
        console.error('Missing critical DOM elements.');
        return;
    }
    const config = {
        uploadUrl,
        sessionId,
        shortDurationMs: shortDurationSeconds * 1000,
        shortVoicedMs: shortVoicedSeconds * 1000,
        voiceEnabled,
        ttsEndpoint,
        ttsMime,
    };
    new RecorderController(elements, config);
}
document.addEventListener('DOMContentLoaded', bootstrap);
