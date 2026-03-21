from typing import List, Dict, Optional
from dataclasses import dataclass, field
import ast
import re


@dataclass
class PageChunk:
    """A semantic unit within a page (function, class, method, import block, etc.)."""
    chunk_id: str
    page_id: str
    parent_id: Optional[str]
    text: str
    start_pos: int
    end_pos: int
    chunk_type: str  # 'function' | 'class' | 'method' | 'import'
    metadata: Dict = field(default_factory=dict)
    children: List[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class Page:
    """Represents a logical page (one source file)."""
    page_id: str
    file_path: str
    language: str
    chunks: Dict[str, PageChunk] = field(default_factory=dict)
    root_chunks: List[str] = field(default_factory=list)
    breadcrumb: List[str] = field(default_factory=list)
    table_of_contents: List[Dict] = field(default_factory=list)


class HierarchicalPageIndexer:
    """Build and manage a hierarchical page index from source files."""

    def __init__(self):
        self.pages: Dict[str, Page] = {}
        self.chunk_counter = 0

    def _generate_chunk_id(self) -> str:
        self.chunk_counter += 1
        return f"chunk_{self.chunk_counter}"

    def index_file(self, file_path: str, content: str, language: str) -> Page:
        """Parse a source file and create its page with hierarchical chunks."""
        page_id = f"page_{len(self.pages)}"
        page = Page(
            page_id=page_id,
            file_path=file_path,
            language=language,
            breadcrumb=[file_path],
        )
        units = self._parse_code(content, language)
        self._build_chunks(page, units, content)
        self.pages[page_id] = page
        return page

    def _build_chunks(self, page: Page, units: List[Dict], content: str):
        """Create chunks from parsed units and wire up parent-child relationships."""
        # First pass: create all chunks
        id_map: Dict[int, str] = {}  # unit index -> chunk_id
        for i, unit in enumerate(units):
            chunk_id = self._generate_chunk_id()
            id_map[i] = chunk_id
            chunk = PageChunk(
                chunk_id=chunk_id,
                page_id=page.page_id,
                parent_id=None,
                text=unit['text'],
                start_pos=unit['start'],
                end_pos=unit['end'],
                chunk_type=unit['type'],
                metadata={
                    'name': unit.get('name'),
                    'language': page.language,
                    'file': page.file_path,
                    'type': unit['type'],
                    'line': unit.get('line', 0),
                },
            )
            page.chunks[chunk_id] = chunk

        # Second pass: wire parent-child from unit['parent_idx']
        for i, unit in enumerate(units):
            chunk_id = id_map[i]
            parent_idx = unit.get('parent_idx')
            if parent_idx is not None and parent_idx in id_map:
                parent_cid = id_map[parent_idx]
                page.chunks[chunk_id].parent_id = parent_cid
                page.chunks[parent_cid].children.append(chunk_id)
            else:
                page.root_chunks.append(chunk_id)

            page.table_of_contents.append({
                'chunk_id': chunk_id,
                'name': unit.get('name'),
                'type': unit['type'],
                'line': unit.get('line', 0),
                'parent': id_map.get(parent_idx) if parent_idx is not None else None,
            })

    # ── Python parsing via AST ──────────────────────────────────────────

    def _parse_code(self, content: str, language: str) -> List[Dict]:
        if language == 'python':
            return self._parse_python(content)
        elif language in ('javascript', 'typescript'):
            return self._parse_js_ts(content)
        return []

    def _parse_python(self, content: str) -> List[Dict]:
        """Use Python's ast module for robust parsing with method extraction."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._parse_python_fallback(content)

        lines = content.split('\n')
        units: List[Dict] = []
        unit_idx_counter = [0]  # mutable counter for assigning indexes

        def _get_source(node: ast.AST) -> str:
            """Extract source text for an AST node."""
            start_line = node.lineno - 1
            end_line = getattr(node, 'end_lineno', node.lineno)
            return '\n'.join(lines[start_line:end_line])

        def _get_pos(node: ast.AST):
            start_line = node.lineno - 1
            start = sum(len(lines[i]) + 1 for i in range(start_line))
            end_line = getattr(node, 'end_lineno', node.lineno)
            end = sum(len(lines[i]) + 1 for i in range(end_line))
            return start, end

        # Gather imports
        import_lines = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_lines.append(lines[node.lineno - 1])
        if import_lines:
            first_import = None
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    first_import = node
                    break
            start, _ = _get_pos(first_import)
            units.append({
                'type': 'import',
                'name': 'imports',
                'text': '\n'.join(import_lines),
                'start': start,
                'end': start + len('\n'.join(import_lines)),
                'line': first_import.lineno,
                'parent_idx': None,
            })

        # Walk top-level definitions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                start, end = _get_pos(node)
                class_idx = len(units)
                units.append({
                    'type': 'class',
                    'name': node.name,
                    'text': _get_source(node),
                    'start': start,
                    'end': end,
                    'line': node.lineno,
                    'parent_idx': None,
                })
                # Extract methods inside the class
                for item in ast.iter_child_nodes(node):
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        m_start, m_end = _get_pos(item)
                        units.append({
                            'type': 'method',
                            'name': item.name,
                            'text': _get_source(item),
                            'start': m_start,
                            'end': m_end,
                            'line': item.lineno,
                            'parent_idx': class_idx,
                        })

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start, end = _get_pos(node)
                units.append({
                    'type': 'function',
                    'name': node.name,
                    'text': _get_source(node),
                    'start': start,
                    'end': end,
                    'line': node.lineno,
                    'parent_idx': None,
                })

        return sorted(units, key=lambda x: x['start'])

    def _parse_python_fallback(self, content: str) -> List[Dict]:
        """Regex fallback for files that fail ast.parse (e.g. syntax errors)."""
        units = []
        for match in re.finditer(r'^class\s+(\w+)', content, re.MULTILINE):
            start = match.start()
            end = self._find_indent_block_end(content, start)
            units.append({
                'type': 'class', 'name': match.group(1),
                'text': content[start:end], 'start': start, 'end': end,
                'line': content[:start].count('\n') + 1, 'parent_idx': None,
            })
        for match in re.finditer(r'^def\s+(\w+)\s*\(', content, re.MULTILINE):
            start = match.start()
            end = self._find_indent_block_end(content, start)
            units.append({
                'type': 'function', 'name': match.group(1),
                'text': content[start:end], 'start': start, 'end': end,
                'line': content[:start].count('\n') + 1, 'parent_idx': None,
            })
        import_matches = list(re.finditer(r'^(?:from|import)\s+.+', content, re.MULTILINE))
        if import_matches:
            text = '\n'.join(m.group(0) for m in import_matches)
            units.append({
                'type': 'import', 'name': 'imports', 'text': text,
                'start': import_matches[0].start(), 'end': import_matches[-1].end(),
                'line': content[:import_matches[0].start()].count('\n') + 1,
                'parent_idx': None,
            })
        return sorted(units, key=lambda x: x['start'])

    # ── JS/TS parsing ───────────────────────────────────────────────────

    def _parse_js_ts(self, content: str) -> List[Dict]:
        """Parse JavaScript/TypeScript with brace-matching (skips strings and comments)."""
        units = []

        # Classes
        for match in re.finditer(r'^(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(\w+)', content, re.MULTILINE):
            start = match.start()
            end = self._find_js_block_end(content, start)
            class_idx = len(units)
            class_text = content[start:end]
            units.append({
                'type': 'class', 'name': match.group(1),
                'text': class_text, 'start': start, 'end': end,
                'line': content[:start].count('\n') + 1, 'parent_idx': None,
            })
            # Extract methods inside the class body
            self._extract_js_methods(class_text, start, class_idx, units)

        # Top-level functions
        for match in re.finditer(
            r'^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)\s*\(|'
            r'^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[a-zA-Z_]\w*)\s*=>',
            content, re.MULTILINE,
        ):
            name = match.group(1) or match.group(2)
            start = match.start()
            end = self._find_js_block_end(content, start)
            # Skip if this range overlaps with a class
            if any(u['type'] == 'class' and u['start'] <= start < u['end'] for u in units):
                continue
            units.append({
                'type': 'function', 'name': name,
                'text': content[start:end], 'start': start, 'end': end,
                'line': content[:start].count('\n') + 1, 'parent_idx': None,
            })

        # Imports
        import_matches = list(re.finditer(r'^import\s+.+', content, re.MULTILINE))
        if import_matches:
            text = '\n'.join(m.group(0) for m in import_matches)
            units.append({
                'type': 'import', 'name': 'imports', 'text': text,
                'start': import_matches[0].start(), 'end': import_matches[-1].end(),
                'line': content[:import_matches[0].start()].count('\n') + 1,
                'parent_idx': None,
            })

        # React components (const X = () => { ... } or function X)
        for match in re.finditer(
            r'^(?:export\s+)?(?:const|let)\s+(\w+)\s*[:=]\s*(?:React\.FC|FC)',
            content, re.MULTILINE,
        ):
            name = match.group(1)
            start = match.start()
            end = self._find_js_block_end(content, start)
            if any(u['name'] == name for u in units):
                continue
            units.append({
                'type': 'function', 'name': name,
                'text': content[start:end], 'start': start, 'end': end,
                'line': content[:start].count('\n') + 1, 'parent_idx': None,
            })

        return sorted(units, key=lambda x: x['start'])

    def _extract_js_methods(self, class_text: str, class_start: int, class_idx: int, units: List[Dict]):
        """Extract methods from a JS/TS class body."""
        # Match method patterns: async? methodName(...) {
        for match in re.finditer(
            r'^\s+(?:async\s+)?(?:static\s+)?(?:get\s+|set\s+)?(\w+)\s*\([^)]*\)\s*(?::\s*\w+(?:<[^>]+>)?\s*)?\{',
            class_text, re.MULTILINE,
        ):
            name = match.group(1)
            if name in ('if', 'for', 'while', 'switch', 'catch'):
                continue
            method_start = class_start + match.start()
            method_end = self._find_js_block_end(class_text, match.start()) + class_start
            units.append({
                'type': 'method', 'name': name,
                'text': class_text[match.start():method_end - class_start],
                'start': method_start, 'end': method_end,
                'line': class_text[:match.start()].count('\n') + 1,
                'parent_idx': class_idx,
            })

    # ── Block boundary helpers ──────────────────────────────────────────

    def _find_indent_block_end(self, content: str, start: int) -> int:
        """Find end of a Python indentation-based block."""
        lines = content[start:].split('\n')
        if len(lines) <= 1:
            return len(content)
        for i in range(1, len(lines)):
            line = lines[i]
            if line and not line[0].isspace() and line.strip() and not line.startswith('#'):
                pos = start
                for j in range(i):
                    pos += len(lines[j]) + 1
                return pos
        return len(content)

    def _find_js_block_end(self, content: str, start: int) -> int:
        """Find end of a JS/TS block by brace-matching, skipping strings and comments."""
        i = start
        depth = 0
        found_open = False
        length = len(content)
        while i < length:
            c = content[i]
            # Skip single-line comments
            if c == '/' and i + 1 < length and content[i + 1] == '/':
                i = content.find('\n', i)
                if i == -1:
                    return length
                i += 1
                continue
            # Skip multi-line comments
            if c == '/' and i + 1 < length and content[i + 1] == '*':
                i = content.find('*/', i + 2)
                if i == -1:
                    return length
                i += 2
                continue
            # Skip strings
            if c in ('"', "'", '`'):
                i = self._skip_js_string(content, i)
                continue
            if c == '{':
                depth += 1
                found_open = True
            elif c == '}':
                depth -= 1
                if found_open and depth == 0:
                    # Include trailing semicolon/newline
                    end = i + 1
                    if end < length and content[end] == ';':
                        end += 1
                    return end
            i += 1
        return length

    @staticmethod
    def _skip_js_string(content: str, start: int) -> int:
        """Skip past a quoted string (handles escape chars)."""
        quote = content[start]
        i = start + 1
        length = len(content)
        while i < length:
            c = content[i]
            if c == '\\':
                i += 2
                continue
            if c == quote:
                return i + 1
            # Template literal newlines are OK
            if quote != '`' and c == '\n':
                return i
            i += 1
        return length

    # ── Context expansion ───────────────────────────────────────────────

    def expand_context(self, chunk_id: str) -> Optional[Dict]:
        """Return the chunk along with its parent and sibling chunks."""
        for page in self.pages.values():
            if chunk_id in page.chunks:
                chunk = page.chunks[chunk_id]
                result = {'target': chunk, 'parent': None, 'siblings': []}
                if chunk.parent_id and chunk.parent_id in page.chunks:
                    parent = page.chunks[chunk.parent_id]
                    result['parent'] = parent
                    for sibling_id in parent.children:
                        if sibling_id != chunk_id:
                            result['siblings'].append(page.chunks[sibling_id])
                else:
                    # Siblings are other root chunks
                    for root_id in page.root_chunks:
                        if root_id != chunk_id:
                            result['siblings'].append(page.chunks[root_id])
                return result
        return None

    def get_breadcrumb(self, chunk_id: str) -> List[str]:
        """Return hierarchical navigation path: [file, class, method]."""
        for page in self.pages.values():
            if chunk_id in page.chunks:
                chunk = page.chunks[chunk_id]
                path = [page.file_path]
                # Walk up parent chain
                current = chunk
                names = []
                while current.parent_id and current.parent_id in page.chunks:
                    parent = page.chunks[current.parent_id]
                    names.append(parent.metadata.get('name') or parent.chunk_type)
                    current = parent
                names.reverse()
                path.extend(names)
                name = chunk.metadata.get('name')
                if name:
                    path.append(name)
                return path
        return []

    def get_page_toc(self, page_id: str) -> List[Dict]:
        page = self.pages.get(page_id)
        return page.table_of_contents if page else []

    # ── Serialization ───────────────────────────────────────────────────

    def serialize(self) -> Dict:
        """Convert index to a JSON-serializable dict."""
        return {
            'chunk_counter': self.chunk_counter,
            'pages': {
                pid: {
                    'file_path': page.file_path,
                    'language': page.language,
                    'breadcrumb': page.breadcrumb,
                    'toc': page.table_of_contents,
                    'root_chunks': page.root_chunks,
                    'chunks': {
                        cid: {
                            'text': chunk.text,
                            'type': chunk.chunk_type,
                            'metadata': chunk.metadata,
                            'parent_id': chunk.parent_id,
                            'children': chunk.children,
                            'start_pos': chunk.start_pos,
                            'end_pos': chunk.end_pos,
                        }
                        for cid, chunk in page.chunks.items()
                    },
                }
                for pid, page in self.pages.items()
            },
        }

    @classmethod
    def deserialize(cls, data: Dict) -> 'HierarchicalPageIndexer':
        """Reconstruct indexer from serialized dict."""
        indexer = cls()
        indexer.chunk_counter = data.get('chunk_counter', 0)
        for pid, pdata in data.get('pages', {}).items():
            page = Page(
                page_id=pid,
                file_path=pdata['file_path'],
                language=pdata['language'],
                breadcrumb=pdata['breadcrumb'],
                table_of_contents=pdata.get('toc', []),
                root_chunks=pdata.get('root_chunks', []),
            )
            for cid, cdata in pdata.get('chunks', {}).items():
                page.chunks[cid] = PageChunk(
                    chunk_id=cid,
                    page_id=pid,
                    parent_id=cdata.get('parent_id'),
                    text=cdata['text'],
                    start_pos=cdata.get('start_pos', 0),
                    end_pos=cdata.get('end_pos', 0),
                    chunk_type=cdata['type'],
                    metadata=cdata.get('metadata', {}),
                    children=cdata.get('children', []),
                )
            indexer.pages[pid] = page
        return indexer
