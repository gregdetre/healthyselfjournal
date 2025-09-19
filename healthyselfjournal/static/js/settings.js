"use strict";
(function () {
    const form = document.getElementById('prefs-form');
    form?.addEventListener('submit', async (ev) => {
        ev.preventDefault();
        const fd = new FormData(form);
        if (!fd.has('resume')) fd.set('resume', 'false'); else fd.set('resume', 'true');
        if (!fd.has('voice')) fd.set('voice', 'false'); else fd.set('voice', 'true');
        await fetch('/settings/save', { method: 'POST', body: fd });
    });

    const btn = document.getElementById('apply-restart');
    btn?.addEventListener('click', async () => {
        const fd = new FormData(form);
        if (!fd.has('resume')) fd.set('resume', 'false'); else fd.set('resume', 'true');
        if (!fd.has('voice')) fd.set('voice', 'false'); else fd.set('voice', 'true');
        await fetch('/settings/save', { method: 'POST', body: fd });
        try { await window.pywebview?.api?.apply_and_restart?.(); } catch { }
        setTimeout(() => location.reload(), 500);
    });

    const pick = document.getElementById('pick-folder');
    pick?.addEventListener('click', async () => {
        try {
            const p = await window.pywebview?.api?.pick_sessions_dir?.();
            if (p) {
                const input = document.getElementById('sessions_dir');
                if (input) input.value = p;
            }
        } catch { }
    });
})();


