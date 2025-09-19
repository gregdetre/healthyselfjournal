"use strict";
(function () {
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


