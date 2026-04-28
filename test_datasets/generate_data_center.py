"""Generate production-level Data Center / Cloud Infrastructure inventory.

Produces two CSVs:
  5_data_center_capacity.csv         (week 1 snapshot)
  5_data_center_capacity_week2.csv   (week 2 — degraded scenario)

Scale: 1500 rows, 23 columns. Multi-region, multi-rack, multi-tier.

Engineered crisis scenarios (deterministic via seed=2026):
  - ~3% nodes near power_budget breach (>95%) → triggers Keeper/Mover
  - ~2% nodes near memory exhaustion (>92%) → triggers Mover (workload move)
  - ~1.5% nodes with status=FAILED → triggers Keeper
  - ~5% nodes overdue maintenance (>180 days) → preventive crisis signal
  - Regional skew: 35% us-east-1, 25% us-west-2, 20% eu-west-1, 12% ap-southeast-1, 8% ap-south-1

Schema fully exercises Sentinel's multi-tenant inferencer:
  - Primary key: server_id
  - Constraint pairs: cores_used<=cores_total, memory_gb_used<=memory_gb_total,
    storage_tb_used<=storage_tb_total, power_watts_current<=power_watts_budget,
    network_gbps_current<=network_gbps_budget
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List


SEED = 2026
N_ROWS = 1500
OUT_DIR = Path(__file__).parent

REGIONS = [
    ("us-east-1", "DC-USE1", 0.35),
    ("us-west-2", "DC-USW2", 0.25),
    ("eu-west-1", "DC-EUW1", 0.20),
    ("ap-southeast-1", "DC-APSE1", 0.12),
    ("ap-south-1", "DC-APS1", 0.08),
]

SERVER_MODELS = [
    # (name, cores_total, memory_gb_total, storage_tb_total, power_budget_w, network_gbps_budget, vendor, monthly_cost)
    ("Dell PowerEdge R750",   64, 512, 30,  800, 50,  "Dell",        1850),
    ("Dell PowerEdge R650",   48, 256, 20,  650, 25,  "Dell",        1450),
    ("HPE ProLiant DL380",    64, 512, 25,  750, 50,  "HPE",         1820),
    ("HPE ProLiant DL360",    32, 192, 15,  550, 25,  "HPE",         1180),
    ("Cisco UCS C240 M6",     48, 384, 24,  720, 40,  "Cisco",       1640),
    ("Lenovo ThinkSystem SR650", 56, 512, 30, 770, 50, "Lenovo",     1720),
    ("Supermicro X12",        96, 768, 48, 1000, 100, "Supermicro",  2400),
    ("Inspur NF5280M6",       64, 512, 30,  820, 50,  "Inspur",      1680),
    ("Dell PowerEdge XE9680", 192, 2048, 60, 1800, 200, "Dell",      6500),  # GPU-heavy
]

CUSTOMER_TIERS = [
    ("STANDARD",   0.60),
    ("PREMIUM",    0.28),
    ("ENTERPRISE", 0.10),
    ("RESERVED",   0.02),
]

STATUS_BASE = [
    ("ACTIVE",      0.92),
    ("MAINTENANCE", 0.04),
    ("DRAINING",    0.02),
    ("FAILED",      0.015),
    ("PROVISIONING",0.005),
]


@dataclass
class Server:
    server_id: str
    hostname: str
    server_model: str
    region: str
    availability_zone: str
    datacenter: str
    rack_id: str
    cpu_cores_used: int
    cpu_cores_total: int
    memory_gb_used: int
    memory_gb_total: int
    storage_tb_used: float
    storage_tb_total: float
    power_watts_current: int
    power_watts_budget: int
    network_gbps_current: int
    network_gbps_budget: int
    status: str
    customer_tier: str
    hardware_age_years: float
    last_maintenance_days_ago: int
    vendor: str
    monthly_cost_usd: int


def weighted_pick(rng: random.Random, choices):
    r = rng.random()
    cum = 0.0
    for item, w in choices:
        cum += w
        if r <= cum:
            return item
    return choices[-1][0]


def generate(seed: int, n_rows: int, week2_degraded: bool = False) -> List[Server]:
    rng = random.Random(seed + (1 if week2_degraded else 0))
    rows: List[Server] = []

    for i in range(n_rows):
        region_pick = weighted_pick(rng, [(r, w) for r, _, w in REGIONS])
        dc_prefix = next(dc for r, dc, _ in REGIONS if r == region_pick)
        az_letter = rng.choice(["a", "b", "c"])
        az = f"{region_pick}{az_letter}"
        datacenter = f"{dc_prefix}{az_letter.upper()}"
        rack_id = f"RACK-{rng.choice(['A','B','C','D','E','F','G','H'])}{rng.randint(1,30):02d}"
        node_num = i + 1

        model_idx = rng.randint(0, len(SERVER_MODELS) - 1)
        model_name, cores_total, mem_total, stor_total, power_budget, net_budget, vendor, monthly_cost = SERVER_MODELS[model_idx]

        # Realistic utilization distributions (skewed by status & tier)
        # Base utilization 30-80%; engineered crisis scenarios push some to 92-99%
        crisis_seed = rng.random()

        if crisis_seed < 0.03:
            # near-power-breach scenario
            cpu_util = rng.uniform(0.85, 0.97)
            mem_util = rng.uniform(0.80, 0.92)
            stor_util = rng.uniform(0.55, 0.75)
            power_util = rng.uniform(0.95, 0.995)
            net_util = rng.uniform(0.60, 0.85)
        elif crisis_seed < 0.05:
            # memory exhaustion scenario
            cpu_util = rng.uniform(0.70, 0.85)
            mem_util = rng.uniform(0.92, 0.99)
            stor_util = rng.uniform(0.65, 0.85)
            power_util = rng.uniform(0.70, 0.85)
            net_util = rng.uniform(0.50, 0.80)
        elif crisis_seed < 0.07:
            # storage near-full scenario
            cpu_util = rng.uniform(0.40, 0.65)
            mem_util = rng.uniform(0.50, 0.70)
            stor_util = rng.uniform(0.93, 0.99)
            power_util = rng.uniform(0.55, 0.75)
            net_util = rng.uniform(0.30, 0.55)
        else:
            # normal load
            cpu_util = rng.betavariate(2.5, 4.0)  # mean ~0.38, right-skew
            mem_util = rng.betavariate(3.0, 3.5)  # mean ~0.46
            stor_util = rng.betavariate(2.0, 3.0)  # mean ~0.40
            power_util = 0.20 + 0.55 * cpu_util + rng.uniform(-0.05, 0.05)  # power follows CPU
            net_util = rng.betavariate(1.8, 5.0)  # mean ~0.27, network typically lower

        # Status — overdue maintenance + small failure rate
        if crisis_seed >= 0.985:
            status = "FAILED"
            cpu_util = mem_util = stor_util = 0.0
            power_util = 0.05  # idle draw
            net_util = 0.0
        else:
            status = weighted_pick(rng, STATUS_BASE)
            if status == "DRAINING":
                cpu_util *= 0.3
                mem_util *= 0.4
                power_util *= 0.6

        # Week 2 degradation: bump utilizations 5-15% on subset
        if week2_degraded and rng.random() < 0.30:
            cpu_util = min(0.99, cpu_util + rng.uniform(0.05, 0.15))
            mem_util = min(0.99, mem_util + rng.uniform(0.05, 0.12))
            power_util = min(0.99, power_util + rng.uniform(0.05, 0.10))

        cpu_cores_used = max(0, min(cores_total, int(round(cpu_util * cores_total))))
        memory_gb_used = max(0, min(mem_total, int(round(mem_util * mem_total))))
        storage_tb_used = round(max(0.0, min(stor_total, stor_util * stor_total)), 2)
        power_watts_current = max(0, min(power_budget, int(round(power_util * power_budget))))
        network_gbps_current = max(0, min(net_budget, int(round(net_util * net_budget))))

        customer_tier = weighted_pick(rng, CUSTOMER_TIERS)

        # Hardware age skewed: most 1-4 years, some up to 7
        hardware_age = round(rng.triangular(0.5, 7.0, 2.5), 2)

        # Maintenance backlog: ~10% overdue (>180 days)
        if rng.random() < 0.10:
            last_maint = rng.randint(180, 540)
        else:
            last_maint = rng.randint(0, 180)

        # Adjust for week2: maintenance ages
        if week2_degraded:
            last_maint = min(last_maint + 7, 720)

        server_id = f"{dc_prefix}-{rack_id}-NODE-{node_num:04d}"
        hostname = f"srv-{region_pick.replace('-', '')}{az_letter}-{node_num:04d}.internal"

        rows.append(Server(
            server_id=server_id,
            hostname=hostname,
            server_model=model_name,
            region=region_pick,
            availability_zone=az,
            datacenter=datacenter,
            rack_id=rack_id,
            cpu_cores_used=cpu_cores_used,
            cpu_cores_total=cores_total,
            memory_gb_used=memory_gb_used,
            memory_gb_total=mem_total,
            storage_tb_used=storage_tb_used,
            storage_tb_total=float(stor_total),
            power_watts_current=power_watts_current,
            power_watts_budget=power_budget,
            network_gbps_current=network_gbps_current,
            network_gbps_budget=net_budget,
            status=status,
            customer_tier=customer_tier,
            hardware_age_years=hardware_age,
            last_maintenance_days_ago=last_maint,
            vendor=vendor,
            monthly_cost_usd=monthly_cost,
        ))

    return rows


def write_csv(rows: List[Server], path: Path) -> None:
    fieldnames = [
        "server_id", "hostname", "server_model", "region", "availability_zone",
        "datacenter", "rack_id", "cpu_cores_used", "cpu_cores_total",
        "memory_gb_used", "memory_gb_total", "storage_tb_used", "storage_tb_total",
        "power_watts_current", "power_watts_budget", "network_gbps_current",
        "network_gbps_budget", "status", "customer_tier", "hardware_age_years",
        "last_maintenance_days_ago", "vendor", "monthly_cost_usd",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main() -> None:
    week1 = generate(seed=SEED, n_rows=N_ROWS, week2_degraded=False)
    week2 = generate(seed=SEED, n_rows=N_ROWS, week2_degraded=True)

    week1_path = OUT_DIR / "5_data_center_capacity.csv"
    week2_path = OUT_DIR / "5_data_center_capacity_week2.csv"

    write_csv(week1, week1_path)
    write_csv(week2, week2_path)

    # Summary stats
    def crisis_count(rows: List[Server]) -> dict:
        power_breach = sum(1 for r in rows if r.power_watts_current >= 0.95 * r.power_watts_budget)
        mem_breach = sum(1 for r in rows if r.memory_gb_total > 0 and r.memory_gb_used >= 0.92 * r.memory_gb_total)
        stor_breach = sum(1 for r in rows if r.storage_tb_total > 0 and r.storage_tb_used >= 0.93 * r.storage_tb_total)
        failed = sum(1 for r in rows if r.status == "FAILED")
        overdue = sum(1 for r in rows if r.last_maintenance_days_ago > 180)
        return {
            "rows": len(rows),
            "power_breach (>=95%)": power_breach,
            "memory_breach (>=92%)": mem_breach,
            "storage_breach (>=93%)": stor_breach,
            "failed_nodes": failed,
            "overdue_maintenance (>180d)": overdue,
        }

    print(f"Wrote {week1_path}")
    for k, v in crisis_count(week1).items():
        print(f"  {k}: {v}")
    print(f"Wrote {week2_path}")
    for k, v in crisis_count(week2).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
