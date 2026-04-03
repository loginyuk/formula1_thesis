import os


def log(summary_lines, *args, **kwargs):
    """
    Prints to terminal and appends to summary_lines for file saving.
    """
    text = " ".join(str(a) for a in args)
    print(text, **kwargs)
    summary_lines.append(text)


def write_summary(summary_lines, path):
    """
    Writes summary_lines to a file, creating parent directories as needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"Summary saved to {path}")
