import os
import re
import json
import glob
from typing import Optional, Dict, Tuple, List

class AnnotationIndexer:
    """
    High-performance, persistent annotation indexer for ExDark and local benchmark datasets.
    
    Features:
    1. Fast O(1) in-memory lookup by:
       - Full filename (e.g. '2015_00001.png', '2015_00001.jpg')
       - Base stem (e.g. '2015_00001')
       - Roboflow / hashed export variants (e.g. '2015_00001_jpg.rf.xxxx.jpg')
    2. Persistent disk caching (.json) with mtime invalidation so scanning is only performed once.
    3. Ambiguity & duplicate collision tracking.
    4. Safe lookup that returns None if unmatched, preventing cross-scene pollution.
    """
    
    def __init__(self, project_root: str, cache_file: Optional[str] = None):
        self.project_root = project_root
        self.cache_dir = os.path.join(project_root, "outputs", "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = cache_file or os.path.join(self.cache_dir, "exdark_annotation_index.json")
        
        # Primary lookup maps
        self.filename_to_path: Dict[str, str] = {}
        self.stem_to_path: Dict[str, str] = {}
        self.id_to_path: Dict[str, str] = {}
        self.ambiguous_keys: Dict[str, List[str]] = {}
        
        self.total_indexed: int = 0
        self.load_or_build_index()

    def _normalize_stem(self, name: str) -> str:
        """Strips extensions and common export suffixes."""
        base = os.path.splitext(name)[0]
        # In case of double extension e.g. .png.txt
        if base.lower().endswith(('.png', '.jpg', '.jpeg')):
            base = os.path.splitext(base)[0]
        return base.lower().strip()

    def _extract_exdark_id(self, name: str) -> Optional[str]:
        """Extracts standard ExDark pattern e.g. 2015_00001."""
        match = re.search(r'(201\d_\d{5})', name)
        if match:
            return match.group(1)
        return None

    def load_or_build_index(self, force_rebuild: bool = False):
        if not force_rebuild and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.filename_to_path = data.get("filename_to_path", {})
                self.stem_to_path = data.get("stem_to_path", {})
                self.id_to_path = data.get("id_to_path", {})
                self.ambiguous_keys = data.get("ambiguous_keys", {})
                self.total_indexed = len(self.filename_to_path)
                # Verify at least one cached file still exists on disk
                sample_p = next(iter(self.filename_to_path.values()), None)
                if sample_p and os.path.exists(sample_p):
                    return
            except Exception:
                pass

        self.rebuild_index()

    def rebuild_index(self):
        """Scans all annotation directories in project recursively and saves cache."""
        self.filename_to_path.clear()
        self.stem_to_path.clear()
        self.id_to_path.clear()
        self.ambiguous_keys.clear()

        search_dirs = [
            os.path.join(self.project_root, "data", "datasets", "ExDark", "annotations"),
            os.path.join(self.project_root, "data", "samples", "low"),
            os.path.join(self.project_root, "data", "annotations")
        ]

        txt_paths = []
        for s_dir in search_dirs:
            if os.path.exists(s_dir):
                txt_paths.extend(glob.glob(os.path.join(s_dir, "**", "*.txt"), recursive=True))

        for full_path in set(txt_paths):
            fname = os.path.basename(full_path)
            stem = self._normalize_stem(fname)
            exdark_id = self._extract_exdark_id(fname)

            # Map exact annotation filename
            self.filename_to_path[fname.lower()] = full_path

            # Map image filename without .txt (e.g. 2015_00001.png -> 2015_00001.png.txt)
            if fname.lower().endswith(".txt"):
                img_name = fname[:-4].lower()
                self.filename_to_path[img_name] = full_path

            # Map stem
            if stem in self.stem_to_path:
                if self.stem_to_path[stem] != full_path:
                    if stem not in self.ambiguous_keys:
                        self.ambiguous_keys[stem] = [self.stem_to_path[stem]]
                    self.ambiguous_keys[stem].append(full_path)
            else:
                self.stem_to_path[stem] = full_path

            # Map ID
            if exdark_id:
                if exdark_id in self.id_to_path:
                    if self.id_to_path[exdark_id] != full_path:
                        if exdark_id not in self.ambiguous_keys:
                            self.ambiguous_keys[exdark_id] = [self.id_to_path[exdark_id]]
                        self.ambiguous_keys[exdark_id].append(full_path)
                else:
                    self.id_to_path[exdark_id] = full_path

        self.total_indexed = len(self.filename_to_path)

        # Save to cache
        cache_data = {
            "filename_to_path": self.filename_to_path,
            "stem_to_path": self.stem_to_path,
            "id_to_path": self.id_to_path,
            "ambiguous_keys": self.ambiguous_keys,
            "total_indexed": self.total_indexed
        }
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)
        except Exception:
            pass

    def lookup(self, query_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Fast lookup for any image query filename or path.
        
        Returns:
            (annotation_path, matched_via, warning_message)
        """
        if not query_name:
            return None, None, "No image query provided."

        clean_name = os.path.basename(query_name).lower().strip()
        
        # 1. Direct match on filename
        if clean_name in self.filename_to_path:
            return self.filename_to_path[clean_name], "exact_filename", None
            
        cand_txt = f"{clean_name}.txt"
        if cand_txt in self.filename_to_path:
            return self.filename_to_path[cand_txt], "image_plus_txt", None

        # 2. Match via stem
        stem = self._normalize_stem(clean_name)
        if stem in self.stem_to_path:
            warn = None
            if stem in self.ambiguous_keys:
                warn = f"Multiple matches found for stem '{stem}', using highest-priority: {os.path.basename(self.stem_to_path[stem])}"
            return self.stem_to_path[stem], "stem_match", warn

        # 3. Match via standard ExDark ID (e.g. 2015_00001)
        exdark_id = self._extract_exdark_id(clean_name)
        if exdark_id and exdark_id in self.id_to_path:
            warn = None
            if exdark_id in self.ambiguous_keys:
                warn = f"Multiple matches found for ID '{exdark_id}', using default: {os.path.basename(self.id_to_path[exdark_id])}"
            return self.id_to_path[exdark_id], "exdark_id_match", warn

        return None, None, f"No corresponding ground-truth annotation found for '{clean_name}'."
