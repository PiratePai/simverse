"""Compatibility entry point for the frozen R27 evidence."""

if __name__ == "__main__":
    if __package__:
        from .replay_frozen import legacy_entry
    else:
        from replay_frozen import legacy_entry

    legacy_entry("r27")
