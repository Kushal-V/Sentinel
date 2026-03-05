"""
generate_test_data.py
=====================
QA Data Generation Pipeline for Sentinel — Supply Chain Digital Twin.

Generates 4 highly realistic industry-specific CSV files to stress-test the
``DynamicSchemaInferencer``. Each dataset uses completely different naming
conventions for its constraint columns, validating that the regex-cascade
detection works across domains.

Usage::

    python generate_test_data.py

Output files are written to ``tests/data/`` by default.

Dependencies::

    pip install faker pandas
    # or: uv add faker
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED: int = 42
HOSPITAL_ROWS: int = 80
SHIPPING_ROWS: int = 60
SERVER_ROWS: int = 75
RETAIL_ROWS: int = 100

OUTPUT_DIR: Path = Path(__file__).resolve().parent / "tests" / "data"

fake = Faker("en_US")
Faker.seed(SEED)
random.seed(SEED)

# ---------------------------------------------------------------------------
# Shared Pools (realistic domain-specific vocabularies)
# ---------------------------------------------------------------------------

HOSPITAL_WARDS = [
    "Cardiology ICU", "Neurology Ward", "Orthopaedics", "Maternity",
    "Oncology", "Paediatrics", "General Surgery", "Emergency", "Burns Unit",
    "Renal Dialysis", "Psychiatric", "Respiratory Care", "Geriatrics",
]
HOSPITAL_DEPARTMENTS = [
    "Medicine", "Surgery", "Obstetrics", "Paediatrics", "Psychiatry",
    "Oncology", "Emergency", "Rehabilitation", "Cardiology", "Neurology",
]

VESSEL_TYPES = ["Container Ship", "Bulk Carrier", "Tanker", "RoRo", "Reefer"]
SHIPPING_ROUTES = [
    "Shanghai → Rotterdam",
    "Singapore → Los Angeles",
    "Hamburg → New York",
    "Busan → Long Beach",
    "Dubai → Felixstowe",
    "Mumbai → Colombo → Singapore",
    "Santos → Rotterdam",
    "Guangzhou → Sydney",
    "Tanjung Pelepas → Le Havre",
    "Algeciras → Baltimore",
]
FLAGS = ["Panama", "Liberia", "Marshall Islands", "Bahamas", "Malta",
         "Singapore", "Hong Kong", "Cyprus", "Greece", "Norway"]

SERVER_TYPES = [
    "GPU Compute (A100)", "CPU General Purpose (Intel Xeon)",
    "Memory Optimised (2TB RAM)", "Storage Dense (PetaByte NAS)",
    "High Frequency Trading Blade", "GPU Compute (H100)",
    "ARM Graviton (Cloud Native)", "FPGA Accelerator",
]
DATACENTRE_LOCATIONS = [
    "US-East-1 (Virginia)", "EU-West-2 (London)", "AP-Southeast-1 (Singapore)",
    "US-West-2 (Oregon)", "EU-Central-1 (Frankfurt)", "AP-Northeast-1 (Tokyo)",
    "CA-Central-1 (Montreal)", "SA-East-1 (São Paulo)",
]

CLOTHING_LINES = [
    "Men's Athletic", "Women's Casual", "Kids' Formal", "Unisex Streetwear",
    "Women's Formal", "Men's Outdoor", "Luxury Evening Wear", "Sustainable Basics",
    "Seasonal Collection", "Kids' Play",
]
BRANDS = ["ArcoWear", "NovaTex", "UrbanCraft", "SwiftFit", "TerraCotton",
          "LuxeBlend", "PureForm", "AtmosWear", "FiberEdge", "GlacierLine"]
MATERIALS = ["100% Cotton", "Merino Wool", "Recycled Polyester", "Linen Blend",
             "Bamboo Fibre", "Gore-Tex", "Cashmere Blend", "Organic Hemp"]

# ---------------------------------------------------------------------------
# 1. Hospital Management
# ---------------------------------------------------------------------------

def generate_hospital_data() -> pd.DataFrame:
    """Generate realistic hospital ward occupancy data.

    Constraint pair: ``currently_admitted`` → ``max_bed_capacity``
    The ``DynamicSchemaInferencer`` must detect: stem="admitted" or "bed",
    limit column contains "max" and "capacity".
    """
    rows: list[dict] = []
    used_ward_ids: set[str] = set()

    for _ in range(HOSPITAL_ROWS):
        # Generate a unique ward_id
        while True:
            dept_code = random.choice(["CARD", "NEUR", "ORTH", "MATN", "ONCO",
                                       "PAED", "SURG", "EMER", "BURN", "RENAL",
                                       "PSYC", "RESP", "GERI"])
            floor = random.randint(1, 8)
            wing = random.choice(["A", "B", "C", "D"])
            ward_id = f"WARD-{dept_code}-{floor}{wing}"
            if ward_id not in used_ward_ids:
                used_ward_ids.add(ward_id)
                break

        max_cap = random.choice([10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 48])
        # Realistic occupancy: 40–100% utilisation with bursts
        occupancy_rate = random.triangular(0.40, 1.10, 0.85)  # triangle dist
        currently = min(int(max_cap * occupancy_rate), max_cap)  # clamp to cap

        # Daily cost: ICU wards are far more expensive
        is_icu = "ICU" in ward_id or dept_code in ("CARD", "BURN", "EMER", "RENAL")
        base_cost = random.uniform(1800, 4200) if is_icu else random.uniform(450, 1200)
        daily_cost = round(base_cost * currently, 2)

        # Average length of stay (days)
        avg_los = round(random.triangular(1.5, 30.0, 5.0), 1)

        rows.append({
            "ward_id": ward_id,
            "department": random.choice(HOSPITAL_DEPARTMENTS),
            "ward_name": random.choice(HOSPITAL_WARDS),
            "currently_admitted": currently,
            "max_bed_capacity": max_cap,
            "avg_length_of_stay_days": avg_los,
            "icu_level": random.randint(1, 3) if is_icu else 0,
            "daily_cost_usd": daily_cost,
            "staff_on_duty": random.randint(2, max(3, currently // 4)),
            "ventilators_in_use": random.randint(0, min(4, currently // 5)) if is_icu else 0,
            "pending_admissions": random.randint(0, 12),
            "contact_consultant": fake.name(),
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["ward_id"]).reset_index(drop=True)
    print(f"  [Hospital]   {len(df):>3} rows | "
          f"avg occupancy: {(df['currently_admitted'] / df['max_bed_capacity']).mean():.1%}")
    return df


# ---------------------------------------------------------------------------
# 2. Global Shipping Fleet
# ---------------------------------------------------------------------------

def generate_shipping_data() -> pd.DataFrame:
    """Generate realistic container shipping fleet data.

    Constraint pair: ``containers_loaded`` → ``teu_limit``
    The inferencer must detect that "loaded" is the mutable value and
    "limit" is the cap using the ``{stem}_limit`` pattern.
    """
    rows: list[dict] = []
    used_vessel_ids: set[str] = set()

    teu_classes = {
        "Ultra Large (24,000 TEU)": (22000, 24000),
        "Very Large (18,000 TEU)":  (15000, 18000),
        "Large (14,000 TEU)":       (12000, 14200),
        "Post-Panamax (10,000 TEU)":(8000,  10000),
        "Panamax (5,000 TEU)":      (4000,   5200),
        "Feeder (1,500 TEU)":       (900,    1800),
    }

    for _ in range(SHIPPING_ROWS):
        # Unique IMO-style vessel ID
        while True:
            vessel_id = f"IMO-{random.randint(9_100_000, 9_999_999)}"
            if vessel_id not in used_vessel_ids:
                used_vessel_ids.add(vessel_id)
                break

        vessel_class = random.choice(list(teu_classes.keys()))
        teu_min, teu_max = teu_classes[vessel_class]
        teu_limit = random.randint(teu_min, teu_max)

        load_factor = random.triangular(0.55, 0.98, 0.82)
        containers_loaded = int(teu_limit * load_factor)

        vessel_name = (
            f"{fake.last_name()} "
            f"{random.choice(['Pioneer', 'Horizon', 'Meridian', 'Express', 'Spirit', 'Star'])}"
        )
        fuel_eff = round(random.uniform(2.5, 8.5), 2)  # grams CO2 per TEU-km

        # Speed knots: inversely correlated with size
        speed_knots = round(random.uniform(14, 24), 1)

        reefer_plugs = random.randint(0, 800)
        reefer_occupied = random.randint(0, reefer_plugs)

        rows.append({
            "vessel_id": vessel_id,
            "vessel_name": vessel_name,
            "vessel_type": random.choice(VESSEL_TYPES),
            "vessel_class": vessel_class,
            "flag_state": random.choice(FLAGS),
            "route": random.choice(SHIPPING_ROUTES),
            "containers_loaded": containers_loaded,
            "teu_limit": teu_limit,
            "reefer_containers": reefer_occupied,
            "reefer_plug_capacity": reefer_plugs,
            "speed_knots": speed_knots,
            "fuel_efficiency_g_co2_per_teu_km": fuel_eff,
            "estimated_arrival_days": random.randint(6, 45),
            "charter_rate_usd_per_day": round(random.uniform(8000, 85000), 0),
            "captain": fake.name(),
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["vessel_id"]).reset_index(drop=True)
    print(f"  [Shipping]   {len(df):>3} rows | "
          f"avg load factor: {(df['containers_loaded'] / df['teu_limit']).mean():.1%}")
    return df


# ---------------------------------------------------------------------------
# 3. Data Centre / Server Farm
# ---------------------------------------------------------------------------

def generate_server_data() -> pd.DataFrame:
    """Generate realistic data centre rack utilisation data.

    Constraint pair: ``active_compute_tb`` → ``max_compute_allowance``
    Tests the inferencer's ``{value}_{suffix} → max_{value}`` pattern where
    the suffix "allowance" is non-standard.
    """
    rows: list[dict] = []
    used_rack_ids: set[str] = set()

    for _ in range(SERVER_ROWS):
        while True:
            dc = random.choice(["US1", "EU2", "AP1", "US3", "EU4", "CA1"])
            aisle = random.choice(["A", "B", "C", "D", "E"])
            rack_num = random.randint(1, 60)
            rack_id = f"RACK-{dc}-{aisle}{rack_num:02d}"
            if rack_id not in used_rack_ids:
                used_rack_ids.add(rack_id)
                break

        server_type = random.choice(SERVER_TYPES)
        dc_location = random.choice(DATACENTRE_LOCATIONS)

        # max compute in TB — varies by rack type
        if "GPU" in server_type:
            max_compute = round(random.uniform(40.0, 200.0), 1)
        elif "Memory" in server_type:
            max_compute = round(random.uniform(80.0, 320.0), 1)
        elif "Storage" in server_type:
            max_compute = round(random.uniform(500.0, 2000.0), 1)
        else:
            max_compute = round(random.uniform(20.0, 100.0), 1)

        util_rate = random.triangular(0.20, 0.99, 0.72)
        active_compute = round(max_compute * util_rate, 2)

        # Power draw in kW
        power_draw_kw = round(random.uniform(3.2, 42.0), 2)
        max_power_kw = round(power_draw_kw / util_rate * 1.15, 2)  # headroom

        rows.append({
            "rack_id": rack_id,
            "data_centre_location": dc_location,
            "server_type": server_type,
            "active_compute_tb": active_compute,
            "max_compute_allowance": max_compute,
            "power_draw_kw": power_draw_kw,
            "max_power_kw": max_power_kw,
            "servers_online": random.randint(1, 32),
            "servers_total": 32,
            "avg_cpu_utilisation_pct": round(util_rate * 100 * random.uniform(0.85, 1.05), 1),
            "avg_ram_utilisation_pct": round(random.uniform(30.0, 98.0), 1),
            "network_throughput_gbps": round(random.uniform(1.0, 400.0), 2),
            "incidents_last_30d": random.randint(0, 5),
            "sla_tier": random.choice(["Gold", "Platinum", "Standard", "Premium"]),
            "support_contact": fake.email(),
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["rack_id"]).reset_index(drop=True)
    print(f"  [ServerFarm] {len(df):>3} rows | "
          f"avg utilisation: {(df['active_compute_tb'] / df['max_compute_allowance']).mean():.1%}")
    return df


# ---------------------------------------------------------------------------
# 4. Retail Apparel Inventory
# ---------------------------------------------------------------------------

def generate_retail_data() -> pd.DataFrame:
    """Generate realistic retail apparel inventory data.

    Constraint pair: ``units_on_floor`` → ``stockroom_limit``
    Tests the suffix-only detection: "on_floor" is the mutable value,
    "limit" is the cap. The stem "units" is stripped via heuristics.
    """
    sizes = ["XS", "S", "M", "L", "XL", "XXL", "One Size"]
    colours = ["Midnight Black", "Arctic White", "Navy", "Slate Grey",
               "Forest Green", "Burgundy", "Coral", "Sand", "Cobalt Blue",
               "Terracotta", "Sage", "Charcoal", "Ivory", "Rust"]
    categories = [
        "T-Shirt", "Hoodie", "Jeans", "Joggers", "Blazer", "Dress",
        "Polo Shirt", "Shorts", "Cardigan", "Jacket", "Skirt", "Jumper",
        "Swimwear", "Activewear Top", "Leggings",
    ]

    rows: list[dict] = []
    used_skus: set[str] = set()

    for _ in range(RETAIL_ROWS):
        while True:
            brand_code = random.choice(["ARW", "NVT", "UCR", "SFT", "TCN",
                                        "LXB", "PRF", "ATM", "FBE", "GLR"])
            cat_code = random.choice(["TS", "HD", "JN", "JG", "BL", "DR",
                                      "PL", "SH", "CD", "JK", "SK", "JM"])
            size_code = random.choice(sizes).replace(" ", "")
            colour_code = f"{random.randint(100, 999)}"
            sku = f"{brand_code}-{cat_code}-{size_code}-{colour_code}"
            if sku not in used_skus:
                used_skus.add(sku)
                break

        category = random.choice(categories)
        size = random.choice(sizes)
        colour = random.choice(colours)
        brand = random.choice(BRANDS)
        material = random.choice(MATERIALS)
        line = random.choice(CLOTHING_LINES)

        # Stockroom limit: how many units the stockroom bay can hold
        stockroom_limit = random.choice([24, 36, 48, 60, 72, 96, 120])
        # Floor units: what's currently on the sales floor
        floor_util = random.triangular(0.05, 1.0, 0.55)
        units_on_floor = int(stockroom_limit * floor_util)

        # Price points by category
        price_map = {
            "Blazer": (89, 340), "Dress": (45, 280), "Jacket": (79, 420),
            "Jumper": (35, 180), "Cardigan": (40, 190), "Leggings": (25, 95),
        }
        lo, hi = price_map.get(category, (15, 120))
        retail_price = round(random.uniform(lo, hi), 2)
        cost_price = round(retail_price * random.uniform(0.28, 0.45), 2)

        days_since_restock = random.randint(0, 90)
        weekly_sales_velocity = round(random.uniform(0.2, 8.5), 1)

        rows.append({
            "sku": sku,
            "brand": brand,
            "clothing_item": f"{brand} {category}",
            "category": category,
            "line": line,
            "size": size,
            "colour": colour,
            "material": material,
            "units_on_floor": units_on_floor,
            "stockroom_limit": stockroom_limit,
            "retail_price_gbp": retail_price,
            "cost_price_gbp": cost_price,
            "gross_margin_pct": round((retail_price - cost_price) / retail_price * 100, 1),
            "days_since_last_restock": days_since_restock,
            "weekly_sales_velocity": weekly_sales_velocity,
            "days_of_stock_remaining": round(
                units_on_floor / max(weekly_sales_velocity / 7, 0.01), 0
            ),
            "store_location": fake.city(),
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["sku"]).reset_index(drop=True)
    print(f"  [Retail]     {len(df):>3} rows | "
          f"avg floor util: {(df['units_on_floor'] / df['stockroom_limit']).mean():.1%}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate all 4 industry datasets and write them to ``tests/data/``."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print("  Sentinel QA — Industry Dataset Generator")
    print(f"{'='*60}")
    print(f"  Output directory: {OUTPUT_DIR}\n")

    datasets: list[tuple[str, pd.DataFrame]] = [
        ("hospital_data.csv",   generate_hospital_data()),
        ("shipping_fleet.csv",  generate_shipping_data()),
        ("server_farm.csv",     generate_server_data()),
        ("retail_apparel.csv",  generate_retail_data()),
    ]

    for filename, df in datasets:
        out_path = OUTPUT_DIR / filename
        df.to_csv(out_path, index=False)
        print(f"  ✅ Written: {out_path}")

    print(f"\n{'='*60}")
    print("  All datasets generated successfully.")
    print(f"  Run verification: python tests/test_edge_cases.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
