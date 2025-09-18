# Questionary Library Reference

Questionary is a Python library for building beautiful, interactive command-line prompts with minimal code. Selected for the healthyselfjournal init wizard based on active maintenance, comprehensive features, and excellent community support.

## See also

- `healthyselfjournal/cli.py` - existing Typer CLI where Questionary will integrate
- `docs/planning/250917c_publish_to_pypi.md` - deployment planning that may benefit from setup wizard
- [Official Questionary Documentation](https://questionary.readthedocs.io/) - comprehensive API reference and examples
- [GitHub Repository](https://github.com/tmbo/questionary) - source code, issues, and examples directory
- [PyPI Package](https://pypi.org/project/questionary/) - installation and version history

## Installation & Setup

### Basic Installation
```bash
pip install questionary
```

### Version Requirements
- **Python**: 3.8+ required (3.6-3.7 dropped in v2.0.0)
- **Current stable**: 2.1.0 (December 2024)
- **Dependencies**: prompt_toolkit 3.0.0+

### Integration with Existing Project
Since healthyselfjournal already uses Typer and Rich, Questionary complements them perfectly:
```python
# Typer for CLI structure
import typer
# Rich for beautiful output
from rich.console import Console
# Questionary for interactive prompts
import questionary
```

## Core Prompt Types

### Text Input
```python
# Basic text input
api_key = questionary.text("Enter your Anthropic API key:").ask()

# With validation
def validate_api_key(text):
    if not text.startswith("sk-"):
        return "API key should start with 'sk-'"
    return True

api_key = questionary.text(
    "Enter your API key:",
    validate=validate_api_key
).ask()

# Password input (masked)
secret = questionary.password("Enter password:").ask()
```

### Selection Menus
```python
# Single selection
privacy_mode = questionary.select(
    "Choose privacy mode:",
    choices=["Standard", "Privacy Mode", "Offline Only"]
).ask()

# With descriptions
choices = [
    questionary.Choice("Standard", value="standard"),
    questionary.Choice("Privacy Mode", value="privacy", disabled="Coming soon"),
    questionary.Choice("Offline Only", value="offline")
]
mode = questionary.select("Select mode:", choices=choices).ask()
```

### Confirmations
```python
# Yes/No prompt
continue_setup = questionary.confirm(
    "Would you like to test audio recording?",
    default=True
).ask()
```

### Multi-Select (Checkbox)
```python
features = questionary.checkbox(
    "Select features to enable:",
    choices=[
        "Audio transcription",
        "Background MP3 conversion",
        "Auto-summaries",
        "Daily reminders"
    ]
).ask()
```

### Path Selection
```python
# File/directory selection with autocomplete
sessions_dir = questionary.path(
    "Select sessions directory:",
    only_directories=True,
    default="./sessions"
).ask()
```

## Integration Patterns with Typer

### Pattern 1: Enhanced Command with Setup Wizard
```python
@app.command()
def init(
    interactive: bool = typer.Option(True, "--interactive/--no-interactive")
):
    """Initialize the journaling app with guided setup."""

    if not interactive:
        # Use Typer's built-in prompts for simple cases
        api_key = typer.prompt("API Key", hide_input=True)
    else:
        # Use Questionary for rich interactive experience
        console.print("[bold]Welcome to ExaminedLife Journal Setup![/bold]")

        # Multi-step wizard
        api_key = questionary.password(
            "Enter your Anthropic API key:"
        ).ask()

        privacy = questionary.select(
            "Privacy settings:",
            choices=["Standard", "Enhanced Privacy", "Offline Mode"]
        ).ask()

        if questionary.confirm("Test audio recording?").ask():
            run_audio_test()
```

### Pattern 2: Conditional Flows
```python
def setup_wizard():
    """Multi-step setup with conditional logic."""

    # Step 1: API Configuration
    has_key = questionary.confirm(
        "Do you have an Anthropic API key?"
    ).ask()

    if has_key:
        api_key = questionary.password("Enter API key:").ask()
        if not validate_api_key(api_key):
            if questionary.confirm("Invalid key. Try again?").ask():
                return setup_wizard()  # Recursive retry
    else:
        console.print("Visit https://console.anthropic.com to get a key")
        if questionary.confirm("Ready to continue?").ask():
            api_key = questionary.password("Enter API key:").ask()

    # Step 2: Feature Selection (conditional on valid API)
    if api_key:
        features = questionary.checkbox(
            "Enable features:",
            choices=["Transcription", "Summaries", "Voice Output"]
        ).ask()
```

### Pattern 3: Progress Feedback with Rich
```python
from rich.progress import Progress, SpinnerColumn, TextColumn

def test_configuration():
    """Test configuration with visual feedback."""

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        # Test API
        task = progress.add_task("Testing API connection...", total=None)
        test_api_connection()
        progress.update(task, completed=True)

        # Test audio if requested
        if questionary.confirm("Test audio recording?").ask():
            task = progress.add_task("Testing microphone...", total=None)
            test_audio_recording()
            progress.update(task, completed=True)
```

## Common Gotchas & Solutions

### 1. Breaking Changes in v2.0.0
**Issue**: Dropped support for Python 3.6-3.7
**Solution**: Ensure Python 3.8+ before upgrading
```python
import sys
if sys.version_info < (3, 8):
    raise RuntimeError("Python 3.8+ required for Questionary 2.x")
```

### 2. Keyboard Interrupt Handling
**Issue**: Ctrl-C raises `KeyboardInterrupt`, not captured by default
**Solution**: Wrap in try/except or use `unsafe_ask()`
```python
try:
    answer = questionary.text("Name:").ask()
except KeyboardInterrupt:
    print("\nSetup cancelled.")
    return

# Or use unsafe_ask() to return None on cancel
answer = questionary.text("Name:").unsafe_ask()
if answer is None:
    print("Cancelled")
```

### 3. Terminal Compatibility
**Issue**: Advanced features may not work in all terminals
**Solution**: Detect terminal capabilities
```python
import os

if os.environ.get("TERM") == "dumb":
    # Fall back to simple input
    name = input("Name: ")
else:
    # Use rich prompts
    name = questionary.text("Name:").ask()
```

### 4. Validation Error Display
**Issue**: Validation errors can be cryptic
**Solution**: Return clear error messages
```python
def validate_path(path):
    p = Path(path)
    if not p.exists():
        return f"Path '{path}' does not exist"
    if not p.is_dir():
        return f"Path '{path}' is not a directory"
    if not os.access(p, os.W_OK):
        return f"No write permission for '{path}'"
    return True
```

### 5. Style Conflicts with Rich
**Issue**: Custom styles may conflict with Rich formatting
**Solution**: Use Questionary's style parameter carefully
```python
from questionary import Style

custom_style = Style([
    ('question', 'fg:#00aa00 bold'),  # Green questions
    ('answer', 'fg:#0000aa'),          # Blue answers
    ('pointer', 'fg:#aa0000 bold'),    # Red pointer
])

answer = questionary.select(
    "Choose:",
    choices=["A", "B", "C"],
    style=custom_style
).ask()
```

## Best Practices

### 1. Always Handle Cancellation
```python
def safe_prompt(prompt_func):
    """Decorator to handle cancellation gracefully."""
    def wrapper(*args, **kwargs):
        try:
            result = prompt_func(*args, **kwargs).ask()
            if result is None:
                raise typer.Abort()
            return result
        except KeyboardInterrupt:
            raise typer.Abort()
    return wrapper

@safe_prompt
def get_api_key():
    return questionary.password("API Key:")
```

### 2. Provide Defaults Where Sensible
```python
sessions_dir = questionary.path(
    "Sessions directory:",
    default=str(Path.home() / ".examinedlife" / "sessions"),
    only_directories=True
).ask()
```

### 3. Use Choice Objects for Complex Options
```python
choices = [
    questionary.Choice(
        title="🎙️  Standard Recording",
        value="standard",
        description="Records audio with default settings"
    ),
    questionary.Choice(
        title="🔒  Privacy Mode",
        value="privacy",
        description="Local processing only, no cloud services"
    ),
    questionary.Choice(
        title="⚡  Quick Note",
        value="quick",
        description="Text-only, no audio recording"
    )
]
```

### 4. Batch Related Prompts
```python
def get_config():
    """Get all configuration in one flow."""
    return questionary.form(
        api_key=questionary.password("API Key:"),
        mode=questionary.select("Mode:", choices=["Standard", "Privacy"]),
        audio=questionary.confirm("Enable audio?", default=True),
    ).ask()
```

### 5. Provide Clear Instructions
```python
questionary.print(
    "Setup Wizard\n"
    "============\n"
    "This wizard will help you configure the journaling app.\n"
    "Press Ctrl-C at any time to cancel.\n",
    style="bold"
)
```

## Testing Strategies

### 1. Mock User Input
```python
# In tests, mock questionary responses
from unittest.mock import patch

@patch('questionary.text')
def test_setup_wizard(mock_text):
    mock_text.return_value.ask.return_value = "test-api-key"
    result = setup_wizard()
    assert result['api_key'] == "test-api-key"
```

### 2. Use Non-Interactive Mode
```python
def get_input(key, interactive=True):
    """Dual-mode input for testing."""
    if not interactive:
        return os.environ.get(f"TEST_{key.upper()}")
    return questionary.text(f"Enter {key}:").ask()
```

## Migration from Click/Typer Prompts

### Before (Typer)
```python
api_key = typer.prompt("API Key", hide_input=True)
confirm = typer.confirm("Continue?")
```

### After (Questionary)
```python
api_key = questionary.password("API Key:").ask()
confirm = questionary.confirm("Continue?").ask()
```

### Advantages
- Better visual presentation
- Arrow-key navigation for selections
- Multi-select support
- Path autocomplete
- Custom validation with clear errors
- Better keyboard interrupt handling

## Performance Considerations

### Startup Time
- Initial import: ~50-100ms
- Per prompt: ~1-5ms overhead vs input()
- Acceptable for interactive CLI tools

### Memory Usage
- Base: ~5-10MB for questionary + prompt_toolkit
- Negligible increase per prompt
- Garbage collected properly after use

## Community & Support

### Resources
- **Documentation**: https://questionary.readthedocs.io/
- **GitHub Issues**: https://github.com/tmbo/questionary/issues
- **Examples**: https://github.com/tmbo/questionary/tree/master/examples
- **Stack Overflow**: Tag `python-questionary` (limited activity)

### Maintenance Status
- **Active**: Regular updates, responsive maintainers
- **Stars**: 1.7k+ on GitHub
- **Forks**: 93+
- **Latest Release**: v2.1.0 (December 2024)
- **Next Release**: Monitor GitHub for v2.2.0 features

### Known Issues (2024-2025)
- #413 (Dec 2024): Terminal resize handling
- #409 (Nov 2024): Unicode character display
- #398 (Sep 2024): Windows path completion
- #396 (Aug 2024): Custom key bindings
- Most issues have workarounds documented

## Future Considerations

### Potential Enhancements
1. **Fuzzy search**: Currently requires exact matches for selections
2. **Async support**: All operations are synchronous
3. **Plugin system**: Limited extensibility compared to prompt_toolkit directly
4. **Theme system**: Styles are basic compared to Rich capabilities

### Alternative Libraries
If Questionary doesn't meet future needs:
- **InquirerPy**: More features but inactive maintenance
- **prompt_toolkit**: Lower level, more control
- **Rich.prompt**: Basic prompts only, but excellent formatting
- **PyInquirer**: Deprecated, use Questionary instead

## Appendix: Quick Reference

```python
# Import
import questionary
from questionary import Choice, Validator, Style, form

# Basic prompts
text = questionary.text("Enter text:").ask()
password = questionary.password("Secret:").ask()
confirm = questionary.confirm("OK?").ask()
select = questionary.select("Choose:", choices=["A", "B"]).ask()
checkbox = questionary.checkbox("Select:", choices=["X", "Y", "Z"]).ask()
path = questionary.path("File:").ask()

# Advanced
questionary.print("Styled text", style="bold italic fg:cyan")
questionary.press_any_key_to_continue().ask()
answers = questionary.form(
    name=questionary.text("Name:"),
    age=questionary.text("Age:", validate=lambda x: x.isdigit())
).ask()

# Error handling
try:
    result = questionary.text("Input:").ask()
except KeyboardInterrupt:
    print("Cancelled")
```

Last updated: September 2025
Based on Questionary v2.1.0