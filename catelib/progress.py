"""Small terminal progress display; no training or experiment state."""
import sys


def progress(label, completed, total, detail='', *, keep=False):
    """Refresh one terminal line; retain completion/failures in redirected logs.

    keep=True leaves an important intermediate message on its own line. Notebook
    and file streams receive plain text only, without terminal escape sequences.
    """
    bar = '#' * (20 * completed // total)
    message = f'  {label} [{bar:<20}] {completed}/{total}'
    if detail:
        message += f' {detail}'
    finished = completed == total
    if sys.stdout.isatty():
        print('\r\033[2K' + message, end='\n' if finished or keep else '', flush=True)
    elif finished or keep:
        print(message, flush=True)
