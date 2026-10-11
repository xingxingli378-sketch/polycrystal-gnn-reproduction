"""Download the authors' dataset and extract regular files safely on Windows."""
import hashlib
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath
root = Path(__file__).parent
archive = root / 'data.tar.gz'
urllib.request.urlretrieve(
    'https://mesomod.weebly.com/uploads/9/3/4/0/93403436/data.tar.gz',
    archive)
print('SHA256:', hashlib.sha256(archive.read_bytes()).hexdigest())
with tarfile.open(archive) as t:
    for member in t.getmembers():
        p = PurePosixPath(member.name)
        if p.is_absolute() or '..' in p.parts or member.issym() or member.islnk():
            raise ValueError('Unsafe archive member: ' + member.name)
        dest = root.joinpath(*p.parts)
        if member.isdir():
            dest.mkdir(parents=True, exist_ok=True)
        elif member.isfile():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(t.extractfile(member).read())
print('Dataset ready.')
