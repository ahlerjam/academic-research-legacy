"""Zentrale pytest-Fixtures fuer die academic-research-Testsuite (Issue #183).

Bisher hatte tests/ kein conftest.py; haeufig genutzte Bausteine (Repo-Root bzw.
scripts/ auf sys.path, temporaere Vault-DB, browser-use-Mock, Beispiel-PDF, TUM-
Bibliotheksprofil) waren in Dutzenden einzelnen test_*.py dupliziert. Diese Datei
zentralisiert sie.

sys.path-Setup: pytest haengt das Repo-Root ohnehin ueber die "rootdir"-Erkennung
an sys.path (tests/ ist ein Package mit __init__.py), Repo-Root-Importe wie
`academic_vault` oder `scripts` (als Package) funktionieren daher bereits ohne
Zutun. Viele Tests importieren jedoch einzelne Module aus scripts/ direkt als
Top-Level-Modul (z.B. `from dedup import deduplicate`), was zusaetzlich
`scripts/` selbst auf sys.path erfordert -- das haben zuvor ~36 test_*.py-Dateien
einzeln per sys.path.insert(...) erledigt. Dieser Block deckt beide Faelle zentral
ab, damit die pro-Datei-Duplikate ersatzlos entfallen koennen.

Skill-spezifische scripts/-Pfade (z.B. skills/<name>/scripts) sind bewusst NICHT
Teil dieses zentralen Setups -- sie bleiben in den jeweiligen test_*.py, da sie
keine generischen Duplikate sind.

Bereitgestellte Fixtures:
  - temp_vault_db        Pfad auf eine frisch initialisierte SQLite-Vault-DB
  - mock_browser_use     MagicMock als Ersatz fuer browser-use-Aufrufe
  - sample_pdf           Pfad auf tests/fixtures/sample_book.pdf
  - library_profile_tum  geparstes config/library-profiles/tum.yaml als dict
  - fake_embedder        deterministischer Offline-Embedder (384d, kein Modell-Download)

Autouse-Fixtures (kein expliziter Import noetig, greifen automatisch):
  - block_real_embedding_backend        blockt echtes e5-Embedding-Modell (#372)
  - block_real_local_reranker_backend   blockt echtes bge-reranker-v2-m3-Modell (#376)
  - isolate_real_academic_paths         Schutzwall um ~/.academic-research (s.u.)
"""

import contextlib
import hashlib
import math
import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Pfade: Repo-Root UND Repo-Root/scripts auf sys.path, damit sowohl
# `academic_vault`/`scripts` (als Package) als auch einzelne Top-Level-Module
# aus scripts/ (z.B. `import dedup`) importierbar sind. Dieser Block war zuvor
# in ~36 test_*.py dupliziert (mit unterschiedlichen Varianten).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

for _p in (REPO_ROOT, SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FIXTURES_DIR = Path(__file__).parent / "fixtures"
LIBRARY_PROFILES_DIR = REPO_ROOT / "config" / "library-profiles"
EVALS_DIR = Path(__file__).parent / "evals"


# ---------------------------------------------------------------------------
# Schutzwall um den ECHTEN ~/.academic-research-Baum
# ---------------------------------------------------------------------------
# Anlass: Datenverlust am 11.08.2026. Ein Testlauf ueberschrieb
# ~/.academic-research/projects/academic-research/vault.db mit tmp-Testdaten
# (1 Paper, 2 Zitate, 1 Notiz, 1 excluded_source weg), ein zweiter schrieb
# echte Tarballs nach ~/.academic-research/snapshots/. Ursache beide Male: der
# Produktionscode faellt fuer Pfade sinnvollerweise auf den Nutzer-Baum zurueck
# (default_db_path(), os.homedir() in den Hooks), und die Tests hatten diesen
# Fallback nicht abgeschnitten.
#
# Der Wall besteht bewusst aus DREI Schichten, weil jede einzelne verrottet,
# sobald jemand eine neue Env-Variable oder einen neuen Schreibpfad einfuehrt:
#
#   1. UMLEITUNG (isolate_real_academic_paths): HOME zeigt je Test in ein
#      tmp-Verzeichnis. Damit wandern ALLE abgeleiteten Defaults mit --
#      Path.home() in Python, os.homedir() in Node, und zwar ohne dass diese
#      Datei jeden einzelnen Pfad kennen muss. Zusaetzlich werden bereits
#      gesetzte Env-Variablen, die in den echten Baum zeigen, in die Sandbox
#      gespiegelt (in einer Claude-Code-Sitzung kann VAULT_DB_PATH auf den
#      Live-Vault zeigen -- die Umleitung von HOME allein reicht dann nicht).
#   2. SPERRE (_protected_write_audit_hook): ein sys.addaudithook lehnt
#      schreibende Operationen auf den echten Baum ab, egal wie der Pfad
#      zustande kam -- auch aus Modulkonstanten, die schon zur Importzeit
#      eingefroren wurden (academic_vault.server._DEFAULT_DB, der db_path-
#      Closure der MCP-Tools). Diese Schicht ist der eigentliche Wall.
#   3. NACHKONTROLLE (Fingerabdruck vor/nach jedem Test): faengt, was die
#      Sperre prinzipbedingt nicht sieht -- Schreibzugriffe aus SUBPROZESSEN
#      (node hooks/pre-compact.mjs, python -c aus der Vault-Bridge). Der
#      betroffene Test faellt namentlich durch, statt dass der Schaden erst
#      Tage spaeter auffaellt.
#
# Ausdrueckliche Ausnahmen (Marker, in dieser Datei registriert):
#   @pytest.mark.real_home
#       Test braucht den ECHTEN HOME (z.B. weil er ~/.academic-research/venv/
#       bin/python oder Anmeldedaten unter ~/.claude aufloest). Schicht 1
#       entfaellt, Schicht 2 und 3 bleiben aktiv -- lesen/ausfuehren ja,
#       schreiben nein.
#   @pytest.mark.allow_real_academic_writes
#       Test DARF in den echten Baum schreiben. Das ist die einzige Hintertuer,
#       sie schaltet Schicht 2 und 3 ab und ist ausschliesslich fuer bewusste
#       Integrationstests gedacht. Im Zweifel: nicht verwenden.
#
# tests/evals/ ist generell ausgenommen (siehe _test_ist_eval): dort laeuft die
# echte claude-CLI, die ihre Anmeldedaten unter dem echten HOME sucht. Die
# Evals laufen nicht in der CI und sind der bewusste Live-Integrationslayer.
# ---------------------------------------------------------------------------

#: Der HOME-Pfad, wie er beim Import dieser conftest.py galt -- also BEVOR
#: irgendeine Fixture ihn umbiegt. Ab hier die einzige Wahrheit ueber "echt".
REAL_HOME = Path(os.path.expanduser("~"))
REAL_ACADEMIC_ROOT = REAL_HOME / ".academic-research"

#: Kennung des pytest-tmp-Baums; Sandbox-Pfade enthalten sie garantiert.
PYTEST_TMP_MARKER = "pytest-of-"

#: Env-Variablen, die auf einen Pfad im Nutzer-Baum zeigen KOENNEN. Gefunden
#: per Grep ueber academic_vault/ und hooks/ nach os.environ/process.env.
#: Unbelegte Variablen bleiben unbelegt -- sonst wuerden Tests brechen, die
#: gerade das Default-Verhalten ohne Env pruefen (z.B.
#: tests/test_vault_db_path_consistency.py). Umgebogen wird nur, was gesetzt
#: ist UND in den geschuetzten Baum zeigt.
SANDBOXED_ENV_VARS = (
    "VAULT_DB_PATH",
    "ACADEMIC_SNAPSHOTS_DIR",
    "ACADEMIC_DECISIONS_LOG",
    "ACADEMIC_NLI_SCAN_SPOOL",
    "ACADEMIC_REINFORCEMENT_STATE",
    "ACADEMIC_PYTHON",
    "VAULT_GUARD_BYPASS_LOG",
    "VAULT_GUARD_BYPASS_REPORT_STATE",
    "VAULT_GUARD_ENV_SWITCH_LOG",
    "VAULT_GUARD_ENV_SWITCH_REPORT_STATE",
    "VAULT_E5_CACHE_DIR",
    "VAULT_RERANK_LOCAL_CACHE_DIR",
)

#: Eintraege im echten HOME, die das Sandbox-HOME als Symlink uebernimmt.
#:
#: Ohne sie zerschneidet die HOME-Umleitung die Toolchain: ``node`` ist auf
#: dieser Maschine ein asdf-Shim, der seine Version ueber ``~/.tool-versions``
#: aufloest und ohne diese Datei mit Exit 126 abbricht ("No version is set for
#: command node"); ``uv run`` wuerde seinen Cache unter dem tmp-HOME neu
#: aufbauen und in jedes Zeitlimit laufen. Die Liste ist bewusst eine
#: ALLOWLIST und keine Vollspiegelung des HOME: was hier nicht steht, ist fuer
#: den Test unerreichbar -- ``.academic-research`` steht nicht darin, und genau
#: darum geht es. Ein Eintrag, der hier fehlt und gebraucht wird, faellt sofort
#: durch einen roten Test auf; einer, der zu viel durchlaesst, faellt nie auf.
TOOLCHAIN_HOME_ENTRIES = (
    ".tool-versions",
    ".asdf",
    ".asdfrc",
    ".cache",
    ".config",
    ".local",
    ".npm",
    ".npmrc",
    ".nvm",
    ".nvmrc",
    ".node-version",
    ".pyenv",
    ".python-version",
    ".rbenv",
    ".volta",
    ".bun",
    ".cargo",
    ".rustup",
)

#: Einziger Eintrag INNERHALB von ~/.academic-research, den das Sandbox-HOME
#: durchreicht: das Setup-venv. Es ist Toolchain, keine Forschungsdaten --
#: hooks/lib/vault-bridge.mjs sucht dort seinen Python-Interpreter
#: (~/.academic-research/venv/bin/python), und tests/helpers/smoke_core.py
#: startet den MCP-Server damit. Ohne diesen Durchgriff verlieren die Hook- und
#: Smoke-Tests ihren einzigen funktionsfaehigen Interpreter, sobald PATH-python3
#: absichtlich unbrauchbar gemacht wird. Geschuetzt bleibt, worum es geht:
#: projects/ (die vault.db) und snapshots/ werden NICHT durchgereicht.
TOOLCHAIN_ACADEMIC_ENTRIES = ("venv",)


class RealPathWriteBlocked(RuntimeError):
    """Ein Test wollte in einen geschuetzten Pfad schreiben.

    Kein Bug im Wall, sondern der Wall bei der Arbeit: der ausloesende Test
    muss seinen Zielpfad ausdruecklich in ein tmp-Verzeichnis legen (db_path=,
    snapshots_dir=, VAULT_DB_PATH, ACADEMIC_SNAPSHOTS_DIR, CLAUDE_PROJECT_DIR).
    """


_PROTECTED_ROOTS: set[Path] = {REAL_ACADEMIC_ROOT}
#: Stack statt Flag, damit verschachtelte Freigaben sich nicht gegenseitig
#: aufheben.
_GUARD_SUSPENSIONS: list[str] = []


def protected_roots() -> frozenset[Path]:
    """Aktuell gesperrte Wurzelverzeichnisse (Lesekopie)."""
    return frozenset(_PROTECTED_ROOTS)


def _normalized(candidate: object) -> Path | None:
    """Kandidat -> absoluter, normalisierter Pfad; ``None``, wenn irrelevant.

    Bewusst OHNE ``resolve()``/``realpath()``: die Funktion laeuft im
    Audit-Hook, und jeder Dateisystemzugriff dort riskiert Rekursion. Relative
    Pfade werden nicht geprueft -- das CWD der Suite ist das Repo, ein
    relativer Pfad kann den Nutzer-Baum also gar nicht treffen.
    """
    if isinstance(candidate, int) or candidate is None:
        return None
    try:
        if isinstance(candidate, bytes):
            candidate = os.fsdecode(candidate)
        text = str(candidate) if isinstance(candidate, os.PathLike) else candidate
        if not isinstance(text, str) or not text.startswith("/"):
            return None
        return Path(os.path.normpath(text))
    except (TypeError, ValueError, UnicodeDecodeError):
        return None


def is_protected_path(candidate: object) -> bool:
    """Liegt ``candidate`` in einer gesperrten Wurzel (oder IST sie)?"""
    path = _normalized(candidate)
    if path is None:
        return False
    for root in _PROTECTED_ROOTS:
        if path == root or root in path.parents:
            return True
    return False


def mirror_into_sandbox(sandbox_home: Path, real_path: object) -> str:
    """Spiegelt einen Pfad aus dem geschuetzten Baum in die Sandbox.

    Die relative Lage bleibt erhalten (``<real>/projects/x/vault.db`` ->
    ``<sandbox>/.academic-research/projects/x/vault.db``), damit Tests, die auf
    Pfadbestandteile schauen, weiterhin Sinnvolles sehen.
    """
    path = _normalized(real_path) or Path(str(real_path))
    try:
        relativ = path.relative_to(REAL_HOME)
    except ValueError:
        relativ = Path(path.name)
    ziel = sandbox_home / relativ
    ziel.parent.mkdir(parents=True, exist_ok=True)
    return str(ziel)


def build_sandbox_home(sandbox_home: Path) -> Path:
    """Legt ein Sandbox-HOME an: leer bis auf die Toolchain-Allowlist.

    ``.academic-research`` wird als LEERES Verzeichnis angelegt (nicht
    verlinkt), damit Code, der es voraussetzt, es vorfindet -- und zwar leer.
    """
    sandbox_home.mkdir(parents=True, exist_ok=True)
    sandbox_academic = sandbox_home / ".academic-research"
    sandbox_academic.mkdir(exist_ok=True)

    def verlinken(quelle: Path, ziel: Path) -> None:
        if ziel.exists() or ziel.is_symlink() or not quelle.exists():
            return
        try:
            ziel.symlink_to(quelle)
        except OSError:
            # Ein fehlender Symlink macht den Wall nicht unsicher, nur unbequem.
            pass

    for name in TOOLCHAIN_HOME_ENTRIES:
        verlinken(REAL_HOME / name, sandbox_home / name)
    for name in TOOLCHAIN_ACADEMIC_ENTRIES:
        verlinken(REAL_ACADEMIC_ROOT / name, sandbox_academic / name)
    return sandbox_home


@contextlib.contextmanager
def protected_root(path):
    """Sperrt ``path`` voruebergehend zusaetzlich (fuer Tests des Walls selbst)."""
    root = Path(path)
    neu = root not in _PROTECTED_ROOTS
    _PROTECTED_ROOTS.add(root)
    try:
        yield root
    finally:
        if neu:
            _PROTECTED_ROOTS.discard(root)


@contextlib.contextmanager
def allow_writes_to_protected_roots(grund: str = "ausdrueckliche Freigabe"):
    """Setzt die Sperre voruebergehend aus. Nur mit gutem Grund benutzen."""
    _GUARD_SUSPENSIONS.append(grund)
    try:
        yield
    finally:
        _GUARD_SUSPENSIONS.pop()


# Audit-Events mit den Indizes der Pfad-Argumente. Quelle: "Audit events table"
# der CPython-Doku. os.replace/os.unlink/Path.write_text tauchen nicht separat
# auf -- sie loesen os.rename/os.remove/open aus.
_WRITE_EVENT_PATH_ARGS: dict[str, tuple[int, ...]] = {
    "open": (0,),
    "os.mkdir": (0,),
    "os.rmdir": (0,),
    "os.remove": (0,),
    # rename/move zerstoeren auch die Quelle -- daher beide Seiten.
    "os.rename": (0, 1),
    "shutil.move": (0, 1),
    # link/symlink dagegen LESEN das Ziel nur; geschrieben wird allein der
    # Linkpfad (Index 1). Sonst liesse sich nicht einmal ein Symlink AUF den
    # geschuetzten Baum anlegen -- genau das tut build_sandbox_home() fuer das
    # Setup-venv.
    "os.link": (1,),
    "os.symlink": (1,),
    "os.truncate": (0,),
    "shutil.copyfile": (1,),
    "shutil.copymode": (1,),
    "shutil.copystat": (1,),
    "shutil.rmtree": (0,),
    "shutil.unpack_archive": (1,),
    # sqlite3 oeffnet immer schreibend und legt fehlende Dateien an -- genau
    # der Weg, auf dem der Live-Vault ueberschrieben wurde (VaultDB.__init__).
    "sqlite3.connect": (0,),
}

_WRITE_OPEN_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def _open_ist_schreibend(args: tuple) -> bool:
    """``open``-Event: Schreibzugriff? Im Zweifel ja (fail-closed).

    ``builtins.open`` liefert den Modus als String, ``os.open`` stattdessen
    ``None`` plus Flags. Ein unbekanntes Muster gilt als Schreibzugriff -- der
    Wall soll lieber einmal zu viel anschlagen als den naechsten Datenverlust
    durchwinken.
    """
    modus = args[1] if len(args) > 1 else None
    if isinstance(modus, str):
        return any(zeichen in modus for zeichen in "wxa+")
    flags = args[2] if len(args) > 2 else None
    if isinstance(flags, int):
        return bool(flags & _WRITE_OPEN_FLAGS)
    return True


def _protected_write_audit_hook(event: str, args: tuple) -> None:
    """sys.addaudithook: blockt Schreibzugriffe auf gesperrte Wurzeln.

    Erster Test ist ein dict-Lookup ueber den Event-Namen; fuer die ganz
    ueberwiegende Mehrheit der Audit-Events (import, exec, compile, ...) endet
    der Hook damit sofort.
    """
    pfad_indizes = _WRITE_EVENT_PATH_ARGS.get(event)
    if pfad_indizes is None or _GUARD_SUSPENSIONS:
        return
    if event == "open" and not _open_ist_schreibend(args):
        return
    for index in pfad_indizes:
        if index >= len(args):
            continue
        if is_protected_path(args[index]):
            raise RealPathWriteBlocked(
                f"Schutzwall (tests/conftest.py): '{event}' wollte in den "
                f"geschuetzten Baum schreiben -> {args[index]!r}. Kein Test darf "
                f"{REAL_ACADEMIC_ROOT} veraendern. Zielpfad ausdruecklich auf "
                f"tmp_path legen (db_path=/snapshots_dir=/VAULT_DB_PATH/"
                f"ACADEMIC_SNAPSHOTS_DIR/CLAUDE_PROJECT_DIR)."
            )


sys.addaudithook(_protected_write_audit_hook)


def academic_tree_fingerprint(root: Path | None = None) -> dict[str, tuple[int, int]]:
    """Billige Signatur (mtime_ns, size) des Nutzer-Baums.

    Die Tiefe folgt dem Schutzbedarf, nicht der Symmetrie -- ein vollstaendiger
    Walk ueber ~/.academic-research kostet ~45 ms und waere je Test zu teuer:

      * ``<root>`` und alles direkt darin  -- neue Top-Level-Dateien/Ordner.
      * ``<root>/projects`` rekursiv       -- hier liegen die vault.db-Dateien.
        Ein Ueberschreiben an Ort und Stelle aendert den Ordner-mtime NICHT, es
        braucht also die Dateiebene (genau der Vorfall vom 11.08.2026).
      * ``<root>/snapshots`` zwei Ebenen   -- Slug-Ordner und Tarballs darin.
        Der Ordner-mtime reicht nicht: seine Aufloesung folgt dem Kernel-Takt
        (ms), mkdir und Datei-Anlage im selben Takt ergeben dieselbe Signatur.

    Nicht abgedeckt sind die Caches ``models/`` und ``venv/`` unterhalb der
    ersten Ebene: dort liegen keine Forschungsdaten.
    """
    basis = REAL_ACADEMIC_ROOT if root is None else Path(root)
    signatur: dict[str, tuple[int, int]] = {}

    def erfassen(pfad: str) -> None:
        try:
            st = os.stat(pfad, follow_symlinks=False)
        except OSError:
            return
        signatur[pfad] = (st.st_mtime_ns, st.st_size)

    def scannen(verzeichnis: Path, tiefe: int) -> None:
        erfassen(str(verzeichnis))
        if tiefe <= 0:
            return
        try:
            eintraege = list(os.scandir(verzeichnis))
        except OSError:
            return
        for eintrag in eintraege:
            erfassen(eintrag.path)
            if tiefe > 1:
                try:
                    ist_ordner = eintrag.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                if ist_ordner:
                    scannen(Path(eintrag.path), tiefe - 1)

    scannen(basis, 1)
    scannen(basis / "projects", 4)
    scannen(basis / "snapshots", 2)
    return signatur


def fingerprint_diff(
    vorher: dict[str, tuple[int, int]], nachher: dict[str, tuple[int, int]]
) -> list[str]:
    """Menschenlesbare Abweichungen zwischen zwei Fingerabdruecken."""
    abweichungen: list[str] = []
    for pfad in sorted(set(vorher) | set(nachher)):
        alt = vorher.get(pfad)
        neu = nachher.get(pfad)
        if alt == neu:
            continue
        if alt is None:
            abweichungen.append(f"NEU:      {pfad}")
        elif neu is None:
            abweichungen.append(f"GELOESCHT:{pfad}")
        else:
            abweichungen.append(f"GEAENDERT:{pfad} {alt} -> {neu}")
    return abweichungen


def _test_ist_eval(request) -> bool:
    """Liegt der Test unter tests/evals/? (siehe Ausnahme im Abschnittskopf)."""
    try:
        pfad = Path(str(request.node.fspath))
    except Exception:
        return False
    return EVALS_DIR == pfad.parent or EVALS_DIR in pfad.parents


def _wall_gilt_fuer(item) -> bool:
    """Unterliegt dieses Testitem der Sperre/Nachkontrolle?"""
    if item.get_closest_marker("allow_real_academic_writes") is not None:
        return False
    try:
        pfad = Path(str(item.fspath))
    except Exception:
        return True
    return not (EVALS_DIR == pfad.parent or EVALS_DIR in pfad.parents)


def _wall_meldung(abweichungen: list[str]) -> str:
    return (
        "Schutzwall (tests/conftest.py): dieser Test hat den ECHTEN Baum "
        f"{REAL_ACADEMIC_ROOT} veraendert -- vermutlich aus einem Subprozess, den "
        "die Schreibsperre prinzipbedingt nicht sieht (node-Hook, python -c der "
        "Vault-Bridge). Dem Subprozess VAULT_DB_PATH, ACADEMIC_SNAPSHOTS_DIR und "
        "CLAUDE_PROJECT_DIR ausdruecklich auf tmp-Pfade setzen.\n  " + "\n  ".join(abweichungen)
    )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    """Nachkontrolle um die Testfunktion herum (Schicht 3).

    Bewusst ein Hook-Wrapper und keine Fixture-Teardown-Pruefung: nur so wird
    daraus ein echter FEHLSCHLAG des Tests statt eines Teardown-Errors neben
    einem gruenen "passed" -- ein Datenverlust darf nicht wie eine Randnotiz
    aussehen.
    """
    if not _wall_gilt_fuer(item):
        return (yield)

    vorher = academic_tree_fingerprint()
    try:
        ergebnis = yield
    except BaseException as fehler:
        abweichungen = fingerprint_diff(vorher, academic_tree_fingerprint())
        if abweichungen:
            item._wall_gemeldet = True
            raise RealPathWriteBlocked(_wall_meldung(abweichungen)) from fehler
        raise
    abweichungen = fingerprint_diff(vorher, academic_tree_fingerprint())
    if abweichungen:
        item._wall_gemeldet = True
        raise RealPathWriteBlocked(_wall_meldung(abweichungen))
    return ergebnis


def pytest_configure(config):
    """Registriert die beiden Opt-out-Marker des Schutzwalls."""
    config.addinivalue_line(
        "markers",
        "real_home: Test braucht den ECHTEN HOME (venv-Python, ~/.claude-Anmeldedaten). "
        "Die HOME-Umleitung entfaellt, die Schreibsperre bleibt aktiv.",
    )
    config.addinivalue_line(
        "markers",
        "allow_real_academic_writes: Test darf ausnahmsweise in den echten "
        "~/.academic-research-Baum schreiben. Einzige Hintertuer des Schutzwalls.",
    )


@pytest.fixture(autouse=True)
def isolate_real_academic_paths(request, tmp_path_factory, monkeypatch):
    """Schicht 1 des Schutzwalls: HOME und Env-Pfade je Test in die Sandbox.

    Siehe den ausfuehrlichen Abschnittskopf oben. Kurz:
      * ohne Marker: HOME zeigt in tmp,
      * ``real_home``: keine Umleitung, Sperre und Nachkontrolle bleiben,
      * ``allow_real_academic_writes``: Sperre und Nachkontrolle aus,
      * tests/evals/: komplett ausgenommen (echte claude-CLI).

    Die Nachkontrolle um die Testfunktion herum macht ``pytest_runtest_call``
    (echter Fehlschlag statt Teardown-Error). Hier bleibt nur das Fenster, das
    der Wrapper nicht sieht: Setup und Teardown der uebrigen Fixtures.
    """
    ist_eval = _test_ist_eval(request)
    darf_schreiben = request.node.get_closest_marker("allow_real_academic_writes") is not None
    braucht_echten_home = request.node.get_closest_marker("real_home") is not None

    if ist_eval or darf_schreiben:
        with allow_writes_to_protected_roots("Marker allow_real_academic_writes / tests/evals"):
            yield
        return

    if not braucht_echten_home:
        # Bewusst tmp_path_factory statt tmp_path: das Sandbox-HOME darf NICHT
        # im tmp_path des Tests liegen. Zahlreiche Tests behandeln tmp_path als
        # ihr eigenes, leeres Arbeitsverzeichnis und pruefen dessen Inhalt
        # (z.B. tests/test_project_bootstrap.py::test_detect_mode_on_empty_dir).
        sandbox_home = build_sandbox_home(tmp_path_factory.mktemp("sandbox-home"))
        monkeypatch.setenv("HOME", str(sandbox_home))
        for name in SANDBOXED_ENV_VARS:
            wert = os.environ.get(name)
            if wert and is_protected_path(wert):
                monkeypatch.setenv(name, mirror_into_sandbox(sandbox_home, wert))

    vorher = academic_tree_fingerprint()
    yield
    abweichungen = fingerprint_diff(vorher, academic_tree_fingerprint())
    # Hat der Wrapper denselben Schaden schon als Fehlschlag gemeldet, hier
    # nicht noch einmal — sonst steht neben dem roten Test ein Teardown-Error
    # mit derselben Aussage.
    if abweichungen and not getattr(request.node, "_wall_gemeldet", False):
        pytest.fail(_wall_meldung(abweichungen), pytrace=False)


# ---------------------------------------------------------------------------
# Vault-DB
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_vault_db(tmp_path):
    """Frisch initialisierte SQLite-Vault-DB in einem Tempdir.

    Liefert den Pfad (str) auf die DB-Datei mit bereits ausgefuehrtem
    `init_schema()`. Faellt auf ein nacktes sqlite-Setup zurueck, falls
    `academic_vault.db` (noch) nicht importierbar ist, damit Tests, die nur
    eine leere DB-Datei brauchen, die Fixture trotzdem nutzen koennen.
    """
    db_path = tmp_path / "vault.db"
    try:
        from academic_vault.db import VaultDB
    except Exception:
        # Fallback: leere DB-Datei ohne Schema (z.B. wenn sqlite-Extensions fehlen)
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        conn.close()
        return str(db_path)

    db = VaultDB(str(db_path))
    db.init_schema()
    return str(db_path)


# ---------------------------------------------------------------------------
# Embedder-Stub (Issue #372)
# ---------------------------------------------------------------------------
class DeterministicEmbedder:
    """Offline-Stand-in fuer ``intfloat/multilingual-e5-small`` (384d).

    Hashing-Bag-of-Words statt neuronalem Modell: gleiche Tokens ergeben
    gleiche Achsen, das Ergebnis ist L2-normalisiert. Damit verhaelt sich der
    Vektorraum lexikalisch-semantisch genug fuer Retrieval-Tests, ohne dass ein
    Modell heruntergeladen werden muss (CI bleibt hermetisch und offline).
    """

    dim = 384

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in re.findall(r"\w+", text.lower()):
            idx = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture
def fake_embedder():
    """Deterministischer 384d-Embedder ohne Netzwerk/Modell-Download (#372)."""
    return DeterministicEmbedder()


@pytest.fixture(autouse=True)
def block_real_embedding_backend(monkeypatch):
    """Verhindert, dass die Suite echte e5-Artefakte laedt (#372, #374).

    Blockiert zwei Ladepfade: die Modellgewichte (``_load_backend_model``, #372)
    und den Tokenizer, den das Chunking fuer exakte Tokenbudgets nutzt
    (``chunking._load_tokenizer``, #374). Ohne den zweiten Guard zoege jeder
    ``chunk_pages``-Aufruf Tokenizer-Dateien von HuggingFace; das Chunking
    faellt stattdessen auf ``approximate_token_count`` zurueck.

    ``sentence-transformers`` ist seit #372 eine harte Dependency, also laeuft
    ``get_embedder()`` in der CI nicht mehr in einen ImportError. Ohne diesen
    Guard wuerde jeder ``add_paper``-Aufruf der Suite ``multilingual-e5-small``
    (~470 MB) von HuggingFace ziehen — die Suite waere netzabhaengig und um
    Groessenordnungen langsamer.

    Gepatcht wird bewusst nur ``_load_backend_model`` (die unterste Schicht):
    Tests, die ihren Embedder ueber ``get_embedder`` bzw. die Parameter von
    ``ingest_paper_embeddings`` injizieren, bleiben unberuehrt, und der
    Degradations-Pfad (``get_embedder() is None``) wird realistisch geprobt.

    Ausnahme: mit ``VAULT_E5_LIVE_TEST=1`` greift der Guard nicht — das ist das
    bestehende Gate des Live-Tests gegen das echte Modell.
    """
    import os

    if os.environ.get("VAULT_E5_LIVE_TEST") == "1":
        yield
        return

    try:
        import academic_vault.embedding_model as em
    except Exception:
        yield
        return

    def _blocked(*_args, **_kwargs):
        raise RuntimeError(
            "Embedding-Backend im Testlauf blockiert (tests/conftest.py). "
            "Tests injizieren den fake_embedder; fuer das echte Modell "
            "VAULT_E5_LIVE_TEST=1 setzen."
        )

    monkeypatch.setattr(em, "_load_backend_model", _blocked)
    em.reset_embedder_cache()

    try:
        import academic_vault.chunking as chunking
    except Exception:
        chunking = None
    if chunking is not None:
        monkeypatch.setattr(chunking, "_load_tokenizer", _blocked)
        chunking.reset_token_counter_cache()

    try:
        yield
    finally:
        em.reset_embedder_cache()
        if chunking is not None:
            chunking.reset_token_counter_cache()


# ---------------------------------------------------------------------------
# Lokaler Reranker-Backend-Guard (Issue #376)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def block_real_local_reranker_backend(monkeypatch):
    """Verhindert, dass die Suite das echte bge-reranker-v2-m3-Modell laedt (#376).

    Seit #376 ruft ``search_papers(..., rerank=True)`` ``apply_reranker`` immer
    auf -- auch ohne Cloud-API-Keys. Seit #714 ist der lokale Reranker per
    Default aktiv (CrossEncoder/sentence_transformers, kein FlagEmbedding
    mehr). Ohne diesen Guard wuerde jeder derartige Testaufruf versuchen,
    ``BAAI/bge-reranker-v2-m3`` von HuggingFace zu laden -- die Suite waere
    netzabhaengig und um Groessenordnungen langsamer.

    Analog ``block_real_embedding_backend``: gepatcht wird bewusst nur
    ``_load_local_reranker_backend`` (die unterste Schicht), Tests, die den
    lokalen Reranker ueber ``_get_local_reranker`` mocken, bleiben unberuehrt.

    Ausnahme: mit ``VAULT_RERANK_LOCAL_LIVE_TEST=1`` greift der Guard nicht —
    das ist das bestehende Gate des Live-Tests gegen das echte Modell.
    """
    import os

    if os.environ.get("VAULT_RERANK_LOCAL_LIVE_TEST") == "1":
        yield
        return

    try:
        import academic_vault.retrieval as retrieval
    except Exception:
        yield
        return

    def _blocked(*_args, **_kwargs):
        raise RuntimeError(
            "Lokaler Reranker im Testlauf blockiert (tests/conftest.py). Tests "
            "mocken _get_local_reranker; fuer das echte Modell "
            "VAULT_RERANK_LOCAL_LIVE_TEST=1 setzen."
        )

    monkeypatch.setattr(retrieval, "_load_local_reranker_backend", _blocked)
    retrieval.reset_local_reranker_cache()
    try:
        yield
    finally:
        retrieval.reset_local_reranker_cache()


# ---------------------------------------------------------------------------
# browser-use-Mock
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_browser_use():
    """MagicMock als Ersatz fuer browser-use-Interaktionen.

    Verhindert echte Browser-Automation in Unit-Tests. `.run(...)` liefert
    standardmaessig einen leeren Erfolgs-String; einzelne Tests koennen
    Rueckgabewerte/Seiteneffekte ueber den Mock konfigurieren.
    """
    mock = MagicMock(name="browser_use")
    mock.run.return_value = ""
    mock.navigate.return_value = ""
    mock.extract.return_value = {}
    return mock


# ---------------------------------------------------------------------------
# Beispiel-PDF
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_pdf():
    """Pfad auf eine kleine, echte Beispiel-PDF (tests/fixtures/sample_book.pdf)."""
    pdf_path = FIXTURES_DIR / "sample_book.pdf"
    if not pdf_path.is_file():
        pytest.skip(f"Beispiel-PDF fehlt: {pdf_path}")
    return pdf_path


# ---------------------------------------------------------------------------
# Bibliotheksprofil TUM
# ---------------------------------------------------------------------------
@pytest.fixture
def library_profile_tum():
    """Geparstes TUM-Bibliotheksprofil (config/library-profiles/tum.yaml) als dict."""
    import yaml

    profile_path = LIBRARY_PROFILES_DIR / "tum.yaml"
    if not profile_path.is_file():
        pytest.skip(f"TUM-Profil fehlt: {profile_path}")
    with open(profile_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
