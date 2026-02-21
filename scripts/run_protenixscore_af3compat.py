#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in {"1", "true", "yes", "y"}:
        return True
    if val in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_seed_spec(value: str | None) -> list[int] | None:
    if not value:
        return None
    seeds: list[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if re.match(r"^-?\d+\s*-\s*-?\d+$", token):
            start_s, end_s = token.split("-", 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            step = 1 if end >= start else -1
            seeds.extend(list(range(start, end + step, step)))
        else:
            seeds.append(int(token))
    return seeds or None


def _resolve_input_dir(input_dir: Path) -> Path:
    if (input_dir / "run_1").exists():
        return input_dir / "run_1"
    return input_dir


def _load_chain_offset_map(offsets_path: Path) -> list[int] | None:
    try:
        payload = json.loads(offsets_path.read_text())
    except Exception:
        return None
    chains = payload.get("chains") if isinstance(payload, dict) else None
    if not chains:
        return None
    mapping: list[int] = []
    for seg in chains:
        try:
            length = int(seg.get("length") or 0)
            start = int(seg.get("start_resseq_B") or 0)
        except Exception:
            return None
        if length <= 0 or start <= 0:
            return None
        for i in range(length):
            mapping.append(start + i)
    return mapping or None


def _renumber_chain_with_offsets(pdb_path: Path, chain_id: str, mapping: list[int]) -> bool:
    try:
        from Bio.PDB import PDBIO, PDBParser
    except Exception:
        return False
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("model", str(pdb_path))
        model = structure[0]
        if chain_id not in model:
            return False
        chain = model[chain_id]
        residues = [r for r in chain if r.id[0] == " "]
        if len(residues) != len(mapping):
            return False
        for idx, res in enumerate(residues, start=1):
            res.id = (" ", int(mapping[idx - 1]), " ")
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(pdb_path))
        return True
    except Exception:
        return False


def _convert_cif_to_pdb(cif_path: Path, pdb_path: Path) -> bool:
    try:
        from Bio.PDB import MMCIFParser, PDBIO
    except Exception:
        return False
    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("model", str(cif_path))
        io = PDBIO()
        io.set_structure(structure)
        io.save(str(pdb_path))
        return True
    except Exception:
        return False


def _infer_staged_defaults(output_dir: Path, num_samples: int | None, model_seeds: str | None) -> tuple[int, str | None]:
    out_name = output_dir.name.lower()
    if num_samples is not None:
        ns = int(num_samples)
    elif "refold" in out_name:
        ns = 5
    elif "round2" in out_name:
        ns = 2
    else:
        ns = 1

    if model_seeds is not None:
        seeds = model_seeds
    elif "refold" in out_name:
        seeds = "0-19"
    else:
        seeds = None

    return ns, seeds


def _copy_or_link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _pick_model_cif(sample_dir: Path, sample_name: str) -> Path | None:
    direct = sample_dir / f"{sample_name}_model.cif"
    if direct.exists():
        return direct
    candidates = sorted(sample_dir.glob("*.cif"))
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pdb_dir", default=None)
    parser.add_argument("--input_pdb", default=None)
    parser.add_argument("--batch_yaml", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--af3score_repo", default=None)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--db_dir", default=None)
    parser.add_argument("--num_jobs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--model_seeds", default=None)
    parser.add_argument("--no_templates", action="store_true")
    parser.add_argument("--use_msa", type=_str2bool, default=True)
    parser.add_argument("--use_template", type=_str2bool, default=True)
    parser.add_argument("--use_rna_msa", type=_str2bool, default=False)
    parser.add_argument("--diffusion_steps", type=int, default=200)
    parser.add_argument("--recycles", type=int, default=10)
    parser.add_argument("--checkpoint_dir", default=os.environ.get("PROTENIX_CHECKPOINT_DIR"))
    parser.add_argument("--model_name", default="protenix_base_default_v0.5.0")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--write_cif_model", type=_str2bool, default=True)
    parser.add_argument("--write_full_confidence", type=_str2bool, default=True)
    parser.add_argument("--write_summary_confidence", type=_str2bool, default=True)
    parser.add_argument("--write_ipsae", type=_str2bool, default=True)
    parser.add_argument("--ipsae_pae_cutoff", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--export_pdb_dir", default=None)
    parser.add_argument("--export_cif_dir", default=None)
    parser.add_argument("--target_offsets_json", default=None)
    parser.add_argument("--target_chain", default="B")
    parser.add_argument("--msa_path", default=None)
    parser.add_argument("--target_chains", default=None)
    parser.add_argument("--target_chain_sequences", default=None)
    parser.add_argument("--target_msa_path", default=None)
    parser.add_argument("--binder_msa_mode", default="single", choices=["single", "none"])
    parser.add_argument("--msa_cache_dir", default=None)
    parser.add_argument("--msa_source", default="none", choices=["none", "colabfold"])
    parser.add_argument("--msa_host", default="https://api.colabfold.com")
    parser.add_argument("--msa_use_env", type=_str2bool, default=True)
    parser.add_argument("--msa_use_filter", type=_str2bool, default=True)
    parser.add_argument("--msa_cache_refresh", type=_str2bool, default=False)
    args = parser.parse_args()

    if args.batch_yaml:
        try:
            import yaml

            payload = yaml.safe_load(Path(args.batch_yaml).read_text()) or {}
        except Exception:
            try:
                payload = json.loads(Path(args.batch_yaml).read_text())
            except Exception:
                payload = {}
        try:
            if isinstance(payload, dict):
                if payload.get("input_pdb_dir") and not args.input_pdb_dir:
                    args.input_pdb_dir = payload.get("input_pdb_dir")
                if payload.get("input_pdb") and not args.input_pdb:
                    args.input_pdb = payload.get("input_pdb")
        except Exception:
            pass

    output_dir = Path(args.output_dir).resolve()
    if args.input_pdb:
        input_pdb = Path(args.input_pdb).resolve()
        if not input_pdb.exists():
            raise FileNotFoundError(f"input_pdb not found: {input_pdb}")
        single_dir = output_dir / "input_pdbs"
        single_dir.mkdir(parents=True, exist_ok=True)
        dst = single_dir / input_pdb.name
        _copy_or_link(input_pdb, dst)
        input_pdb_dir = single_dir
    else:
        if not args.input_pdb_dir:
            raise RuntimeError("--input_pdb_dir is required when --input_pdb is not provided")
        input_pdb_dir = Path(args.input_pdb_dir).resolve()

    input_pdb_dir = _resolve_input_dir(input_pdb_dir)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    af3score_out_dir = output_dir / "af3score_outputs"
    af3score_out_dir.mkdir(parents=True, exist_ok=True)

    num_samples, model_seeds = _infer_staged_defaults(
        output_dir=output_dir,
        num_samples=args.num_samples,
        model_seeds=args.model_seeds,
    )
    if model_seeds is not None:
        parsed = _parse_seed_spec(model_seeds)
        if not parsed:
            raise ValueError(f"Invalid model_seeds: {model_seeds}")

    try:
        from protenixscore.engine import run_predict_refold
    except ModuleNotFoundError:
        import sys

        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from engine import run_predict_refold  # type: ignore

    run_args = argparse.Namespace(
        input=str(input_pdb_dir),
        output=str(af3score_out_dir),
        recursive=True,
        glob=["*.pdb", "*.cif"],
        use_msa=bool(args.use_msa),
        use_esm=False,
        use_template=False if args.no_templates else bool(args.use_template),
        use_rna_msa=bool(args.use_rna_msa),
        no_templates=bool(args.no_templates),
        msa_path=args.msa_path,
        chain_sequence=[],
        target_chains=args.target_chains,
        target_chain_sequences=args.target_chain_sequences,
        target_msa_path=args.target_msa_path,
        binder_msa_mode=args.binder_msa_mode,
        msa_cache_dir=args.msa_cache_dir,
        msa_source=args.msa_source,
        msa_host=args.msa_host,
        msa_use_env=bool(args.msa_use_env),
        msa_use_filter=bool(args.msa_use_filter),
        msa_cache_refresh=bool(args.msa_cache_refresh),
        convert_pdb_to_cif=True,
        keep_intermediate=False,
        intermediate_dir=None,
        assembly_id=None,
        altloc="first",
        checkpoint_dir=args.checkpoint_dir,
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        num_workers=args.num_workers,
        num_samples=num_samples,
        model_seeds=model_seeds,
        diffusion_steps=args.diffusion_steps,
        recycles=args.recycles,
        batch_size=1,
        max_tokens=None,
        max_atoms=None,
        write_full_confidence=bool(args.write_full_confidence),
        write_summary_confidence=bool(args.write_summary_confidence),
        write_ipsae=bool(args.write_ipsae),
        ipsae_pae_cutoff=args.ipsae_pae_cutoff,
        write_cif_model=bool(args.write_cif_model),
        summary_format="json",
        aggregate_csv=str(output_dir / "metrics.csv"),
        overwrite=bool(args.overwrite),
        failed_log=str(output_dir / "failed_records.txt"),
        missing_atom_policy="reference",
        predict_mode=True,
        score_only=False,
    )
    run_predict_refold(run_args)

    offset_map = None
    if args.target_offsets_json:
        offsets_path = Path(args.target_offsets_json).resolve()
        if offsets_path.exists():
            offset_map = _load_chain_offset_map(offsets_path)

    if args.export_cif_dir:
        export_cif_dir = Path(args.export_cif_dir).resolve()
        export_cif_dir.mkdir(parents=True, exist_ok=True)
        for sample_dir in sorted(af3score_out_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            sample_name = sample_dir.name
            model_cif = _pick_model_cif(sample_dir, sample_name)
            if model_cif is None:
                continue
            _copy_or_link(model_cif, export_cif_dir / f"{sample_name}.cif")

    if args.export_pdb_dir:
        export_pdb_dir = Path(args.export_pdb_dir).resolve()
        export_pdb_dir.mkdir(parents=True, exist_ok=True)
        for sample_dir in sorted(af3score_out_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            sample_name = sample_dir.name
            model_cif = _pick_model_cif(sample_dir, sample_name)
            if model_cif is None:
                continue
            pdb_path = export_pdb_dir / f"{sample_name}.pdb"
            if not pdb_path.exists() and _convert_cif_to_pdb(model_cif, pdb_path) and offset_map:
                _renumber_chain_with_offsets(pdb_path, args.target_chain, offset_map)


if __name__ == "__main__":
    main()
