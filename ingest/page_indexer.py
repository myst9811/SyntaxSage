from typing import List, Dict, Optional
from dataclasses import dataclass, field
import re
import json


@dataclass
class PageChunk:
    """A semantic unit within a page (function, class, import block, etc.)."""
    chunk_id: str
    page_id: str
    parent_id: Optional[str]
    text: str
    start_pos: int
    end_pos: int
    chunk_type: str  # 'function' | 'class' | 'import' | 'docstring'
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
        """Parse a source file and create its page + chunks."""
        page_id = f"page_{len(self.pages)}"
        page = Page(
            page_id=page_id,
            file_path=file_path,
            language=language,
            breadcrumb=[file_path],
        )
        units = self._parse_code(content, language)
        for unit in units:
            chunk_id = self._generate_chunk_id()
            chunk = PageChunk(
                chunk_id=chunk_id,
                page_id=page_id,
                parent_id=None,
                text=unit['text'],
                start_pos=unit['start'],
                end_pos=unit['end'],
                chunk_type=unit['type'],
                metadata={
                    'name': unit.get('name'),
                    'language': language,
                    'file': file_path,
                    'type': unit['type'],
                    'line': unit.get('line', 0),
                },
            )
            page.chunks[chunk_id] = chunk
            page.root_chunks.append(chunk_id)
            page.table_of_contents.append({
                'chunk_id': chunk_id,
                'name': unit.get('name'),
                'type': unit['type'],
                'line': unit.get('line', 0),
            })
        self.pages[page_id] = page
        return page

    def _parse_code(self, content: str, language: str) -> List[Dict]:
        """Extract semantic units from source code using regex."""
        units = []
        if language == 'python':
            lines = content.split('\n')

            # Classes
            for match in re.finditer(r'^class\s+(\w+)', content, re.MULTILINE):
                start = match.start()
                # Extract class body: everything until next top-level class/def or EOF
                end = self._find_block_end(content, start)
                units.append({
                    'type': 'class',
                    'name': match.group(1),
                    'text': content[start:end],
                    'start': start,
                    'end': end,
                    'line': content[:start].count('\n') + 1,
                })

            # Top-level functions
            for match in re.finditer(r'^def\s+(\w+)\s*\(', content, re.MULTILINE):
                start = match.start()
                end = self._find_block_end(content, start)
                units.append({
                    'type': 'function',
                    'name': match.group(1),
                    'text': content[start:end],
                    'start': start,
                    'end': end,
                    'line': content[:start].count('\n') + 1,
                })

            # Import blocks (group consecutive imports together)
            import_matches = list(re.finditer(r'^(?:from|import)\s+.+', content, re.MULTILINE))
            if import_matches:
                text = '\n'.join(m.group(0) for m in import_matches)
                units.append({
                    'type': 'import',
                    'name': 'imports',
                    'text': text,
                    'start': import_matches[0].start(),
                    'end': import_matches[-1].end(),
                    'line': content[:import_matches[0].start()].count('\n') + 1,
                })

        return sorted(units, key=lambda x: x['start'])

    def _find_block_end(self, content: str, start: int) -> int:
        """Find the end of an indented block starting at `start`."""
        lines = content[start:].split('\n')
        if len(lines) <= 1:
            return len(content)
        # First line is the def/class line; find next top-level definition
        for i in range(1, len(lines)):
            line = lines[i]
            if line and not line[0].isspace() and line.strip() and not line.startswith('#'):
                # Another top-level item
                pos = start
                for j in range(i):
                    pos += len(lines[j]) + 1  # +1 for newline
                return pos
        return len(content)

    def expand_context(self, chunk_id: str) -> Optional[Dict]:
        """Return the chunk along with its parent and sibling chunks."""
        for page in self.pages.values():
            if chunk_id in page.chunks:
                chunk = page.chunks[chunk_id]
                result = {'target': chunk, 'parent': None, 'siblings': []}
                if chunk.parent_id:
                    result['parent'] = page.chunks.get(chunk.parent_id)
                    parent = page.chunks[chunk.parent_id]
                    for sibling_id in parent.children:
                        if sibling_id != chunk_id:
                            result['siblings'].append(page.chunks[sibling_id])
                return result
        return None

    def get_breadcrumb(self, chunk_id: str) -> List[str]:
        """Return the navigation path to the chunk's page."""
        for page in self.pages.values():
            if chunk_id in page.chunks:
                return page.breadcrumb
        return []

    def get_page_toc(self, page_id: str) -> List[Dict]:
        page = self.pages.get(page_id)
        return page.table_of_contents if page else []

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
