"""Registre des outils autorisés.

Un agent ne peut appeler un outil que s'il est (1) enregistré, (2) activé
dans la configuration et (3) dans la liste de l'agent. Tout le reste est
refusé. Chaque outil déclare son schéma d'entrée, sa catégorie, son niveau
de risque et ses limites. Les arguments sont validés avant l'exécution.

Approbation humaine : un outil peut exiger une confirmation (ex. écraser un
fichier existant). L'appel est alors suspendu (``ApprovalRequired``), la
mission attend, et l'utilisateur approuve ou refuse depuis le tableau de bord
ou la ligne de commande. Une approbation vaut pour CET appel exact (outil +
arguments) et ne sert qu'une fois : rejouer l'étape ne refait pas l'action.

Il n'y a volontairement aucun outil qui lance des commandes système, et
aucun agent ne peut modifier ses propres permissions.
"""

import ast
import datetime as dt
import ipaddress
import math
import operator
import re
import socket
import threading
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

MAX_OUTPUT = 6000
RISK_LEVELS = {"lecture": "faible", "calcul": "faible", "mémoire": "faible", "réseau": "moyen",
               "écriture": "moyen", "sensible": "élevé"}


class ToolError(Exception):
    pass


class ApprovalRequired(Exception):
    """L'action attend une décision humaine (``approval_id`` en base)."""

    def __init__(self, approval_id: int, tool: str, reason: str):
        super().__init__(f"approbation #{approval_id} requise pour {tool} : {reason}")
        self.approval_id = approval_id
        self.tool = tool
        self.reason = reason


@dataclass
class ToolContext:
    """Ce que le registre sait de l'appel en cours (mission, politique, limites)."""
    mission_id: int | None = None
    step_id: int | None = None
    store: object = None
    approval_mode: str = "sensible"
    max_web_calls: int = 15
    agent: str = ""


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, str]          # nom -> description
    func: Callable[..., str]
    risk: str = "lecture"           # lecture | calcul | mémoire | réseau | écriture | sensible
    required: tuple[str, ...] = field(default_factory=tuple)
    category: str = "données"
    types: dict[str, type] = field(default_factory=dict)   # nom -> type attendu (str par défaut)
    limits: str = ""
    # Renvoie un motif si l'appel doit être approuvé par un humain, sinon None.
    approval: Callable[[dict, str], str | None] | None = None
    network: bool = False
    wants_context: bool = False     # la fonction reçoit ctx=ToolContext

    def needs_approval(self, args: dict, mode: str) -> str | None:
        if mode == "jamais":
            return None
        if self.approval is not None:
            reason = self.approval(args, mode)
            if reason:
                return reason
        if self.risk == "sensible":
            return f"outil à risque élevé ({self.name})"
        if mode == "ecriture" and self.risk in ("écriture", "réseau"):
            return f"politique « ecriture » : {self.risk} via {self.name}"
        return None


class ToolRegistry:
    def __init__(self, disabled: set[str] | None = None):
        self._tools: dict[str, Tool] = {}
        self.disabled = set(disabled or ())
        self._lock = threading.Lock()

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def enabled(self, names) -> list[Tool]:
        return [self._tools[n] for n in names if n in self._tools and n not in self.disabled]

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def catalog(self, names) -> str:
        lines = []
        for t in self.enabled(names):
            params = ", ".join(f'"{k}": {v}' for k, v in t.params.items()) or "aucun"
            extra = f" Limites : {t.limits}" if t.limits else ""
            lines.append(f"- {t.name} : {t.description} Arguments : {{{params}}}{extra}")
        return "\n".join(lines) or "(aucun outil)"

    def describe(self) -> list[dict]:
        return [{"name": t.name, "category": t.category, "risk": t.risk, "level": RISK_LEVELS.get(t.risk, "?"),
                 "params": t.params, "required": list(t.required), "limits": t.limits,
                 "enabled": t.name not in self.disabled, "description": t.description} for t in self.all()]

    def _validate(self, tool: Tool, args) -> dict:
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise ToolError("les arguments doivent être un objet JSON")
        unknown = set(args) - set(tool.params)
        if unknown:
            raise ToolError(f"arguments inconnus : {', '.join(sorted(unknown))}")
        missing = [p for p in tool.required if p not in args]
        if missing:
            raise ToolError(f"arguments manquants : {', '.join(missing)}")
        clean = {}
        for k, v in args.items():
            want = tool.types.get(k, str)
            if want is float and isinstance(v, int) and not isinstance(v, bool):
                v = float(v)
            if want is str and isinstance(v, (int, float)) and not isinstance(v, bool):
                v = str(v)
            if not isinstance(v, want) or isinstance(v, bool) and want is not bool:
                raise ToolError(f"argument « {k} » : {want.__name__} attendu")
            clean[k] = v
        return clean

    def call(self, name: str, args, allowed, ctx: ToolContext | None = None) -> str:
        if name not in self._tools:
            raise ToolError(f"outil inconnu : {name}")
        if name in self.disabled:
            raise ToolError(f"outil désactivé : {name}")
        if name not in allowed:
            raise ToolError(f"outil non autorisé pour cet agent : {name}")
        tool = self._tools[name]
        args = self._validate(tool, args)
        store = getattr(ctx, "store", None)

        if tool.network and ctx is not None and store is not None and ctx.mission_id is not None:
            with self._lock:
                used = store.count_tool_uses(ctx.mission_id, name)
            if used >= ctx.max_web_calls:
                raise ToolError(f"limite atteinte : {ctx.max_web_calls} accès réseau par mission")

        reason = tool.needs_approval(args, ctx.approval_mode if ctx else "sensible")
        if reason:
            if store is None or ctx.mission_id is None:
                raise ToolError(f"action refusée : approbation humaine requise ({reason})")
            prior = store.find_approval(ctx.mission_id, name, args)
            if prior and prior["status"] == "denied":
                raise ToolError("action refusée par l'utilisateur ; continue sans elle ou propose une autre voie")
            if not (prior and prior["status"] == "approved" and store.consume_approval(prior["id"])):
                aid = store.request_approval(ctx.mission_id, ctx.step_id, name, args, reason)
                raise ApprovalRequired(aid, name, reason)

        kwargs = dict(args)
        if tool.wants_context:
            kwargs["ctx"] = ctx
        try:
            out = tool.func(**kwargs)
        except (ToolError, ApprovalRequired):
            raise
        except Exception as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc
        out = str(out)
        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT] + f"\n…[tronqué, {len(out)} caractères au total]"
        return out


# --- outils intégrés ---------------------------------------------------

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCS = {n: getattr(math, n) for n in ("sqrt", "log", "log10", "exp", "sin", "cos", "tan", "floor", "ceil")}
_FUNCS.update(abs=abs, round=round, min=min, max=max)
_CONSTS = {"pi": math.pi, "e": math.e}


def calculate(expression: str) -> str:
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ToolError("exposant trop grand")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS \
                and not node.keywords:
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ToolError("expression non autorisée")

    if len(expression) > 500:
        raise ToolError("expression trop longue")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ToolError("expression invalide (utilise le point pour les décimales)")
    try:
        return str(ev(tree))
    except ToolError as exc:
        if "^" in expression:
            raise ToolError(f"{exc} : pour une puissance, écris 2**10 (le ^ n'est pas une puissance)")
        raise
    except ZeroDivisionError:
        raise ToolError("division par zéro")


def now() -> str:
    return dt.datetime.now().astimezone().strftime("%A %d %B %Y, %H:%M:%S (%Z)")


# Fichiers jamais lisibles ni modifiables par un agent, même dans le workspace.
_PROTECTED = re.compile(r"(^|[\\/])(\.env[^\\/]*|\.git|.*\.(pem|key|p12|pfx|db|sqlite3?)|id_rsa[^\\/]*|"
                        r"secrets?\.(toml|json|ya?ml))([\\/]|$)", re.I)


class Workspace:
    """Lecture/écriture limitées à un seul dossier."""

    MAX_BYTES = 500_000

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, rel: str) -> Path:
        if not isinstance(rel, str) or "\x00" in rel:
            raise ToolError("chemin invalide")
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError("chemin hors du workspace")
        if _PROTECTED.search(str(p.relative_to(self.root))):
            raise ToolError("fichier protégé (secrets, configuration ou base de données)")
        return p

    def read(self, path: str) -> str:
        p = self._path(path)
        if not p.is_file():
            raise ToolError(f"fichier introuvable : {path}")
        if p.stat().st_size > self.MAX_BYTES:
            raise ToolError("fichier trop gros (500 Ko max)")
        return p.read_text(encoding="utf-8", errors="replace")

    def write(self, path: str, content: str) -> str:
        p = self._path(path)
        if len(content) > self.MAX_BYTES:
            raise ToolError("contenu trop gros (500 Ko max)")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".jarvis-tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(p)  # écriture atomique : jamais de fichier à moitié écrit
        return f"écrit : {p.relative_to(self.root).as_posix()} ({len(content)} caractères)"

    def overwrite_reason(self, args: dict, mode: str) -> str | None:
        """Écraser un fichier existant avec un contenu différent est destructeur."""
        try:
            p = self._path(args.get("path", ""))
        except ToolError:
            return None  # l'appel échouera de toute façon, inutile de demander
        if p.is_file():
            try:
                same = p.read_text(encoding="utf-8", errors="replace") == args.get("content")
            except OSError:
                same = False
            if not same:
                return f"écrase le fichier existant « {args.get('path')} »"
        return None

    def exists(self, path: str) -> bool:
        try:
            return self._path(path).is_file()
        except ToolError:
            return False

    def list(self, path: str = ".") -> str:
        p = self._path(path)
        if not p.is_dir():
            raise ToolError(f"dossier introuvable : {path}")
        items = [i for i in sorted(p.iterdir()) if not i.name.endswith(".jarvis-tmp")][:200]
        return "\n".join(f"{'[d]' if i.is_dir() else '   '} {i.relative_to(self.root).as_posix()}"
                         for i in items) or "(vide)"


def analyze_code(source: str, filename: str = "code") -> str:
    """Analyse statique (sans exécution) : structure, imports, complexité, points d'attention."""
    lines = source.splitlines()
    todo = sum(1 for l in lines if re.search(r"\b(TODO|FIXME|XXX)\b", l))
    head = f"{filename} : {len(lines)} lignes, {todo} TODO/FIXME"
    if not filename.endswith(".py"):
        return head + "\n(analyse détaillée disponible pour les fichiers Python uniquement)"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return head + f"\nERREUR DE SYNTAXE ligne {exc.lineno} : {exc.msg}"
    imports, defs, warnings = set(), [], []
    branch_types = (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.BoolOp, ast.IfExp, ast.comprehension)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or ".")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec"):
            warnings.append(f"ligne {node.lineno} : appel à {node.func.id}() (dangereux)")
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            warnings.append(f"ligne {node.lineno} : « except: » nu (avale toutes les erreurs)")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            kind = "classe" if isinstance(node, ast.ClassDef) else "fonction"
            size = (getattr(node, "end_lineno", node.lineno) or node.lineno) - node.lineno + 1
            cx = 1 + sum(isinstance(n, branch_types) for n in ast.walk(node))
            doc = "doc" if ast.get_docstring(node) else "sans doc"
            defs.append(f"- {kind} {node.name} (ligne {node.lineno}, {size} lignes, complexité {cx}, {doc})")
            if kind == "fonction" and cx > 12:
                warnings.append(f"{node.name} : complexité élevée ({cx}), à découper")
    out = [head, f"Imports : {', '.join(sorted(imports)) or 'aucun'}", "Définitions :", *(defs or ["(aucune)"])]
    if warnings:
        out += ["Points d'attention :", *warnings]
    return "\n".join(out)


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip())


def _public_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        raise ToolError(f"hôte introuvable : {host}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            return False
    return True


def fetch_url(url: str) -> str:
    import httpx

    current = url
    for _ in range(4):  # suit au plus 3 redirections, en revérifiant l'hôte à chaque fois
        parsed = urlparse(current)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ToolError("seules les URL http(s) publiques sont autorisées")
        if not _public_host(parsed.hostname):
            raise ToolError("adresse privée ou locale refusée")
        with httpx.Client(timeout=15, follow_redirects=False,
                          headers={"User-Agent": "Jarvis/0.2"}) as client:
            with client.stream("GET", current) as resp:
                if resp.is_redirect:
                    current = str(resp.url.join(resp.headers.get("location", "")))
                    continue
                body = b""
                for chunk in resp.iter_bytes():
                    body += chunk
                    if len(body) > 1_000_000:
                        break
                ctype = resp.headers.get("content-type", "")
                status = resp.status_code
        text = body.decode("utf-8", errors="replace")
        if "html" in ctype:
            ex = _TextExtractor()
            ex.feed(text)
            text = "\n".join(ex.parts)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return f"[HTTP {status}] {current}\n{text}"
    raise ToolError("trop de redirections")



AGENT_MEMORY_KINDS = ("knowledge", "hypothesis")


def build_default_registry(cfg, store=None) -> ToolRegistry:
    reg = ToolRegistry(disabled=cfg.disabled_tools)
    ws = Workspace(cfg.workspace)
    reg.register(Tool("calculatrice", "Calcule une expression mathématique.",
                      {"expression": "texte, ex. '(12*3)/4'"}, calculate, risk="calcul", category="calcul",
                      required=("expression",), limits="500 caractères, pas de code"))
    reg.register(Tool("heure", "Donne la date et l'heure actuelles.", {}, now, category="calcul"))
    reg.register(Tool("lire_fichier", "Lit un fichier texte du workspace.",
                      {"path": "chemin relatif"}, ws.read, required=("path",), category="fichiers",
                      limits="workspace uniquement, 500 Ko, fichiers de secrets interdits"))
    reg.register(Tool("lister_fichiers", "Liste un dossier du workspace.",
                      {"path": "chemin relatif (défaut '.')"}, ws.list, category="fichiers",
                      limits="workspace uniquement, 200 entrées"))
    reg.register(Tool("ecrire_fichier", "Écrit (ou remplace) un fichier texte dans le workspace.",
                      {"path": "chemin relatif", "content": "texte"}, ws.write, risk="écriture",
                      category="fichiers", required=("path", "content"), approval=ws.overwrite_reason,
                      limits="workspace uniquement, 500 Ko ; écraser un fichier existant demande une approbation"))
    reg.register(Tool("lire_page_web", "Télécharge une page web publique et renvoie son texte.",
                      {"url": "URL http(s)"}, fetch_url, risk="réseau", category="web", required=("url",),
                      network=True, limits=f"{cfg.max_web_calls} pages par mission, 1 Mo, adresses privées "
                                           "refusées"))

    def code_tool(path: str) -> str:
        return analyze_code(ws.read(path), path)

    reg.register(Tool("analyser_code", "Analyse statique d'un fichier de code du workspace (structure, "
                      "imports, complexité, erreurs de syntaxe). N'exécute rien.",
                      {"path": "chemin relatif"}, code_tool, required=("path",), category="code",
                      limits="workspace uniquement, aucune exécution"))
    if store is not None:
        def search_memory(query: str) -> str:
            hits = store.search(query, limit=8)
            if not hits:
                return "Rien de pertinent en mémoire."
            out = []
            for h in hits:
                if h["type"] == "mission":
                    out.append(f"- [mission #{h['mission_id']}] {h['objective']} → {h['result'][:300]}")
                else:
                    src = f" (source : {h['source']})" if h.get("source") else ""
                    out.append(f"- [{h['type']}, confiance {h.get('confidence', 1):.1f}] {h['text']}{src}")
            return "\n".join(out)

        reg.register(Tool("memoire_missions", "Cherche dans les missions passées et la mémoire (résultats "
                          "vérifiés, leçons, connaissances, hypothèses).", {"query": "mots-clés"}, search_memory,
                          required=("query",), risk="lecture", category="mémoire"))

        def memorize(kind: str, text: str, source: str = "", confidence: float = 0.7, ctx=None) -> str:
            if kind not in AGENT_MEMORY_KINDS:
                raise ToolError(f"type refusé : un agent ne peut mémoriser que {', '.join(AGENT_MEMORY_KINDS)}")
            if kind == "knowledge" and not source.strip():
                raise ToolError("une connaissance doit citer sa source (sinon, utilise 'hypothesis')")
            mid = getattr(ctx, "mission_id", None)
            new = store.remember(kind, text, source=source or None, mission_id=mid, confidence=confidence)
            return "mémorisé" if new else "déjà en mémoire (doublon ignoré)"

        reg.register(Tool("memoriser", "Mémorise une connaissance (avec sa source) ou une hypothèse pour les "
                          "missions futures. Jamais de secrets ni de données personnelles.",
                          {"kind": "'knowledge' ou 'hypothesis'", "text": "l'information",
                           "source": "URL ou fichier d'origine", "confidence": "0 à 1"},
                          memorize, risk="mémoire", category="mémoire", required=("kind", "text"),
                          types={"confidence": float}, wants_context=True,
                          limits="secrets, e-mails et téléphones masqués ; doublons ignorés"))
    return reg
