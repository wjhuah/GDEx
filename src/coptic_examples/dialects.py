"""Config-driven dialect identity; unknown never defaults to Sahidic."""
import fnmatch
import hashlib
import tomllib
from pathlib import Path
from .config import PROJECT
from .text import norm


class Dialects:
    def __init__(self,path=None):
        self.path=Path(path or PROJECT/'dialects.toml')
        self.data=tomllib.loads(self.path.read_text(encoding='utf-8'))['dialects']
        self.aliases={}
        for key,r in self.data.items():
            for alias in [key,r['name'],*r.get('aliases',[])]:
                normalized=norm(alias).casefold()
                if normalized in self.aliases and self.aliases[normalized]!=key:raise ValueError('Dialect aliases must not overlap: '+alias)
                self.aliases[normalized]=key
        self.codes={code:key for key,r in self.data.items() for code in r.get('codes',[])}
        if len(self.codes)!=sum(len(x.get('codes',[])) for x in self.data.values()):raise ValueError('Dialect codes must be unique')

    def require(self,value):
        key=self.aliases.get(norm(value).casefold())
        if key is None:raise ValueError(f'Unknown/unregistered dialect {value!r}. Choose: '+', '.join(self.data))
        return key

    def identify(self,raw,path):
        key=self.aliases.get(norm(raw).casefold())
        paths={k for k,r in self.data.items() if any(fnmatch.fnmatch(path,g) for g in r.get('path_globs',[]))}
        if key:
            return key,('metadata_path_conflict' if paths and key not in paths else 'metadata_alias')
        # A nonempty unrecognized label is positive evidence of a different or
        # mixed dialect. It must not be overridden by a broad folder fallback.
        if norm(raw):return 'unknown','unrecognized_metadata'
        if len(paths)==1:return next(iter(paths)),'source_path_fallback'
        return 'unknown','missing_or_ambiguous_dialect'

    def from_codes(self,codes):
        return sorted({self.codes[c] for c in codes if c in self.codes})

    @property
    def fingerprint(self):return hashlib.sha256(self.path.read_bytes()).hexdigest()
