#!/usr/bin/env python3
"""Exact computation of the balanced edge-sum span beta(G) using Gurobi.

The model uses binary assignment variables x[v,k], integer edge-sum variables,
and integer lower/upper variables L and U.  A value is exact only when Gurobi
returns OPTIMAL.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class GraphInstance:
    name: str
    vertices: list[str]
    edges: list[tuple[int, int]]

    @property
    def order(self) -> int:
        return len(self.vertices)

    @property
    def size(self) -> int:
        return len(self.edges)


def _edge_list(vertices: list[str], raw_edges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    edge_set: set[tuple[int, int]] = set()
    n = len(vertices)
    for u, v in raw_edges:
        if u == v:
            continue
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u}, {v}) is outside the vertex set")
        a, b = sorted((u, v))
        edge_set.add((a, b))
    return sorted(edge_set)


def path_graph(n: int) -> GraphInstance:
    if n < 2:
        raise ValueError("P_n requires n >= 2")
    vertices = [str(i + 1) for i in range(n)]
    edges = _edge_list(vertices, ((i, i + 1) for i in range(n - 1)))
    return GraphInstance(f"P_{n}", vertices, edges)


def cycle_graph(n: int) -> GraphInstance:
    if n < 3:
        raise ValueError("C_n requires n >= 3")
    vertices = [str(i + 1) for i in range(n)]
    edges = _edge_list(vertices, ((i, (i + 1) % n) for i in range(n)))
    return GraphInstance(f"C_{n}", vertices, edges)


def grid_graph(m: int, n: int) -> GraphInstance:
    if m < 1 or n < 1:
        raise ValueError("P_m square P_n requires m,n >= 1")
    vertices = [f"({i + 1},{j + 1})" for i in range(m) for j in range(n)]

    def idx(i: int, j: int) -> int:
        return i * n + j

    raw_edges: list[tuple[int, int]] = []
    for i in range(m):
        for j in range(n):
            if i + 1 < m:
                raw_edges.append((idx(i, j), idx(i + 1, j)))
            if j + 1 < n:
                raw_edges.append((idx(i, j), idx(i, j + 1)))
    return GraphInstance(f"P_{m}xP_{n}", vertices, _edge_list(vertices, raw_edges))


def torus_graph(m: int, n: int) -> GraphInstance:
    if m < 3 or n < 3:
        raise ValueError("C_m square C_n requires m,n >= 3 for simple cycles")
    vertices = [f"({i + 1},{j + 1})" for i in range(m) for j in range(n)]

    def idx(i: int, j: int) -> int:
        return (i % m) * n + (j % n)

    raw_edges: list[tuple[int, int]] = []
    for i in range(m):
        for j in range(n):
            raw_edges.append((idx(i, j), idx(i + 1, j)))
            raw_edges.append((idx(i, j), idx(i, j + 1)))
    return GraphInstance(f"C_{m}xC_{n}", vertices, _edge_list(vertices, raw_edges))


def build_graph(family: str, dims: list[int]) -> GraphInstance:
    family = family.lower()
    if family in {"path", "p"}:
        if len(dims) != 1:
            raise ValueError("path requires one integer: n")
        return path_graph(dims[0])
    if family in {"cycle", "c"}:
        if len(dims) != 1:
            raise ValueError("cycle requires one integer: n")
        return cycle_graph(dims[0])
    if family in {"grid", "rect", "pgrid"}:
        if len(dims) != 2:
            raise ValueError("grid requires two integers: m n")
        return grid_graph(dims[0], dims[1])
    if family in {"torus", "t", "tgrid"}:
        if len(dims) != 2:
            raise ValueError("torus requires two integers: m n")
        return torus_graph(dims[0], dims[1])
    raise ValueError(f"unknown graph family: {family}")


def parse_case(text: str) -> tuple[str, list[int]]:
    parts = text.replace("x", ":").split(":")
    if len(parts) < 2:
        raise ValueError(f"case must look like grid:3:4, torus:4:4, path:8; got {text!r}")
    return parts[0], [int(p) for p in parts[1:]]


def solve_beta(
    graph: GraphInstance,
    *,
    time_limit: float | None = None,
    threads: int | None = None,
    mip_gap: float | None = None,
    symmetry_break: bool = True,
    log_file: str | None = None,
    write_model: str | None = None,
    quiet: bool = False,
) -> dict:
    if graph.size == 0:
        raise ValueError("beta(G) is defined here only for graphs with at least one edge")

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError(
            "Could not import gurobipy. Activate the Python environment that has Gurobi installed."
        ) from exc

    n = graph.order
    labels = range(1, n + 1)
    vertices = range(n)

    model = gp.Model(f"beta_{graph.name}")
    if quiet:
        model.Params.OutputFlag = 0
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if threads is not None:
        model.Params.Threads = threads
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap
    if log_file is not None:
        model.Params.LogFile = log_file

    x = model.addVars(vertices, labels, vtype=GRB.BINARY, name="x")
    s = model.addVars(range(graph.size), vtype=GRB.INTEGER, lb=3, ub=2 * n - 1, name="s")
    lower = model.addVar(vtype=GRB.INTEGER, lb=3, ub=2 * n - 1, name="L")
    upper = model.addVar(vtype=GRB.INTEGER, lb=3, ub=2 * n - 1, name="U")
    span = model.addVar(vtype=GRB.INTEGER, lb=0, ub=2 * n - 4, name="C")

    model.addConstrs((gp.quicksum(x[v, k] for k in labels) == 1 for v in vertices), name="one_label")
    model.addConstrs((gp.quicksum(x[v, k] for v in vertices) == 1 for k in labels), name="use_label")

    label_expr = {
        v: gp.quicksum(k * x[v, k] for k in labels)
        for v in vertices
    }

    for e_id, (u, v) in enumerate(graph.edges):
        model.addConstr(s[e_id] == label_expr[u] + label_expr[v], name=f"edge_sum[{e_id}]")
        model.addConstr(lower <= s[e_id], name=f"lower[{e_id}]")
        model.addConstr(s[e_id] <= upper, name=f"upper[{e_id}]")

    model.addConstr(span == upper - lower, name="span")
    if symmetry_break and n >= 2:
        model.addConstr(label_expr[0] <= (n + 1) // 2, name="complement_symmetry")

    model.setObjective(span, GRB.MINIMIZE)

    if write_model is not None:
        model.write(write_model)

    model.optimize()

    status_name = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.MEM_LIMIT: "MEM_LIMIT",
    }.get(model.Status, str(model.Status))

    result = {
        "graph": graph.name,
        "order": graph.order,
        "size": graph.size,
        "status": status_name,
        "runtime": model.Runtime,
        "mip_gap": None,
        "best_bound": None,
        "beta": None,
        "L": None,
        "U": None,
        "labels": None,
        "edge_sums": None,
    }

    if model.SolCount > 0:
        beta = int(round(span.X))
        assigned = {}
        for v in vertices:
            for k in labels:
                if x[v, k].X > 0.5:
                    assigned[graph.vertices[v]] = k
                    break
        edge_sums = []
        for e_id, (u, v) in enumerate(graph.edges):
            edge_sums.append({
                "edge": [graph.vertices[u], graph.vertices[v]],
                "sum": int(round(s[e_id].X)),
            })
        result.update({
            "beta": beta,
            "L": int(round(lower.X)),
            "U": int(round(upper.X)),
            "labels": assigned,
            "edge_sums": edge_sums,
            "mip_gap": model.MIPGap if math.isfinite(model.MIPGap) else None,
        })

    if hasattr(model, "ObjBound") and math.isfinite(model.ObjBound):
        result["best_bound"] = math.ceil(model.ObjBound - 1e-9)

    return result


def write_json(result: dict, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def append_csv(rows: list[dict], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "graph",
        "order",
        "size",
        "status",
        "beta",
        "best_bound",
        "L",
        "U",
        "runtime",
        "mip_gap",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def print_summary(result: dict) -> None:
    beta_text = "none" if result["beta"] is None else str(result["beta"])
    bound_text = "none" if result["best_bound"] is None else str(result["best_bound"])
    print(
        f"{result['graph']}: status={result['status']}, beta={beta_text}, "
        f"best_bound={bound_text}, order={result['order']}, size={result['size']}, "
        f"runtime={result['runtime']:.2f}s"
    )


def add_solver_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--time-limit", type=float, default=None, help="Gurobi time limit in seconds")
    parser.add_argument("--threads", type=int, default=None, help="number of Gurobi threads")
    parser.add_argument("--mip-gap", type=float, default=None, help="relative MIP gap tolerance")
    parser.add_argument("--no-symmetry", action="store_true", help="disable complement symmetry breaking")
    parser.add_argument("--quiet", action="store_true", help="suppress Gurobi log output")
    parser.add_argument("--log-file", default=None, help="write Gurobi log to this file")
    parser.add_argument("--write-model", default=None, help="write the LP/MPS model before solving")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute beta(G) exactly with Gurobi.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("solve", help="solve one graph instance")
    single.add_argument("family", choices=["path", "cycle", "grid", "torus", "p", "c", "rect", "t"])
    single.add_argument("dims", nargs="+", type=int)
    single.add_argument("--json", default=None, help="write full solution to JSON")
    add_solver_options(single)

    batch = subparsers.add_parser("batch", help="solve several graph instances")
    batch.add_argument(
        "--cases",
        nargs="+",
        required=True,
        help="cases such as path:8 cycle:8 grid:3:4 torus:4:4",
    )
    batch.add_argument("--csv", default="results/beta_results.csv", help="write batch summary CSV")
    batch.add_argument("--json-dir", default=None, help="write one JSON solution per case")
    add_solver_options(batch)

    args = parser.parse_args()

    solver_kwargs = {
        "time_limit": args.time_limit,
        "threads": args.threads,
        "mip_gap": args.mip_gap,
        "symmetry_break": not args.no_symmetry,
        "log_file": args.log_file,
        "write_model": args.write_model,
        "quiet": args.quiet,
    }

    if args.command == "solve":
        graph = build_graph(args.family, args.dims)
        result = solve_beta(graph, **solver_kwargs)
        print_summary(result)
        if args.json is not None:
            write_json(result, args.json)
        return

    rows = []
    for case in args.cases:
        family, dims = parse_case(case)
        graph = build_graph(family, dims)
        result = solve_beta(graph, **solver_kwargs)
        print_summary(result)
        rows.append(result)
        if args.json_dir is not None:
            json_path = Path(args.json_dir) / f"{graph.name}.json"
            write_json(result, str(json_path))
    append_csv(rows, args.csv)


if __name__ == "__main__":
    main()
