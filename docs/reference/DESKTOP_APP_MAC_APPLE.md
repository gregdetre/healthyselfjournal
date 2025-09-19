## Desktop App (macOS) — Signing, Hardened Runtime, Notarization

This evergreen guide explains how to sign, notarize, and verify the macOS desktop app bundle produced by PyInstaller for Healthy Self Journal. It also lists the required entitlements (including network client for the localhost loopback) and provides step‑by‑step commands.

### See also

- `SETUP_DEV.md` – Development environment, local packaging commands, CI notes
- `DESKTOP_APP_PYWEBVIEW.md` – Desktop shell architecture, security posture, CLI flags
- `PYPI_PUBLISHING.md` – Python package release process (orthogonal to desktop bundling)

Authoritative Apple docs to consult when anything changes:
- Notarization: `https://developer.apple.com/documentation/security/notarizing_your_app_before_distribution`
- Notary tool: `xcrun notarytool` usage: `man notarytool` or Developer docs
- Stapling: `xcrun stapler staple` doc: `man stapler`
- Entitlements reference: `https://developer.apple.com/documentation/bundleresources/entitlements`
- Code signing and hardened runtime: `codesign --help` and Apple Developer docs

### Prerequisites

- Apple Developer Program membership
- Xcode + Command Line Tools installed
- Developer ID Application certificate installed in Login keychain
- PyInstaller build completed (app bundle under `./dist/`)

Relevant files in this repo:
- `packaging/HealthySelfJournal.spec` – PyInstaller spec (references Info.plist and entitlements)
- `packaging/macos/Info.plist` – Bundle metadata; includes mic usage strings
- `packaging/macos/entitlements.plist` – App Sandbox + microphone + network client

### Entitlements (required)

We use App Sandbox with microphone and network‑client to allow WKWebView to reach the local FastHTML server on 127.0.0.1:

```xml
<key>com.apple.security.app-sandbox</key><true/>
<key>com.apple.security.device.audio-input</key><true/>
<key>com.apple.security.network.client</key><true/>
```

File lives at `packaging/macos/entitlements.plist` and is referenced by the PyInstaller spec for the bundle.

### One‑time setup (credentials)

1) Create an App Store Connect API key and download the `.p8` file (note Key ID and Issuer ID). See Apple docs: App Store Connect API keys.
2) Store a notary profile locally so `notarytool` can use it non‑interactively:

```bash
xcrun notarytool store-credentials HSJNotary \
  --key /ABSOLUTE/PATH/TO/AuthKey_ABC12345.p8 \
  --key-id ABC12345 \
  --issuer YOUR-ISSUER-UUID
```

3) Find your Developer ID Application identity:

```bash
security find-identity -v -p codesigning | grep "Developer ID Application"
```

### Build (PyInstaller)

See `SETUP_DEV.md` for environment setup. Local build:

```bash
uv sync --active
uv run --active pyinstaller packaging/HealthySelfJournal.spec
```

Outputs under `dist/`, e.g. `dist/Healthy Self Journal.app`.

### Codesign (hardened runtime)

```bash
APP="/Users/greg/Dropbox/dev/experim/healthyselfjournal/dist/Healthy Self Journal.app"
ENT="/Users/greg/Dropbox/dev/experim/healthyselfjournal/packaging/macos/entitlements.plist"
CERT="Developer ID Application: Your Name (TEAMID)"   # replace with your identity

codesign --force --deep --options runtime --timestamp \
  --entitlements "$ENT" --sign "$CERT" "$APP"

codesign --verify --strict --deep --verbose=2 "$APP"
```

Notes:
- Use `--options runtime` to enable the hardened runtime (required for notarization).
- If verification complains about nested items (Frameworks, dylibs), sign nested code first or rely on `--deep` for development; prefer explicit nested signing for release.

### Notarize and staple

Zip the app (preferred for submission) or build a DMG and submit that.

Zip + submit + staple:

```bash
ZIP="/Users/greg/Dropbox/dev/experim/healthyselfjournal/dist/HealthySelfJournal.zip"
/usr/bin/ditto -c -k --keepParent "$APP" "$ZIP"

xcrun notarytool submit "$ZIP" --keychain-profile HSJNotary --wait

xcrun stapler staple -v "$APP"
spctl -a -vv --type exec "$APP"
```

Optional DMG flow:

```bash
DMG="/Users/greg/Dropbox/dev/experim/healthyselfjournal/dist/HealthySelfJournal.dmg"
hdiutil create -volname "Healthy Self Journal" -srcfolder "$APP" -ov -format UDZO "$DMG"
codesign --force --timestamp --sign "$CERT" "$DMG"
xcrun notarytool submit "$DMG" --keychain-profile HSJNotary --wait
xcrun stapler staple -v "$DMG"
```

### Verification checklist

- `codesign --verify --strict --deep --verbose=2 "$APP"` passes
- `xcrun notarytool history --keychain-profile HSJNotary` shows success (optional)
- `xcrun stapler validate "$APP"` passes
- `spctl -a -vv --type exec "$APP"` shows "accepted"

### Where to find things in this repo

- Desktop architecture, endpoints, CSP, and behavior: `DESKTOP_APP_PYWEBVIEW.md`
- Dev setup and local packaging commands: `SETUP_DEV.md`
- Python packaging to PyPI (CLI distribution, not the desktop app): `PYPI_PUBLISHING.md`
- macOS packaging inputs: `packaging/macos/Info.plist`, `packaging/macos/entitlements.plist`, `packaging/HealthySelfJournal.spec`

### Troubleshooting

- Blank window or mic prompt missing in packaged app: ensure Info.plist contains `NSMicrophoneUsageDescription`; verify entitlements were applied.
- WKWebView cannot reach `127.0.0.1`: confirm `com.apple.security.network.client` is present and the app is signed with that entitlements file.
- Notarization fails: inspect `notarytool log` from the submission result; common issues include missing hardened runtime, unsigned nested code, or quarantine flags on input.
- Gatekeeper blocks: staple the ticket and re-verify with `spctl` on a different machine or a new user account.

### Status

Current state: manual local signing/notarization verified in development. CI automation for desktop artifacts is planned.


