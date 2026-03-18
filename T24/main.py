import httpx
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, MultiPolygon, mapping
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="LumiGuard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static", html=True), name="static")


# ── Polígono da cidade (pontos extremos fornecidos) ───────────────────────────

_EXTREMOS = [
    (-22.255883, -45.714998),
    (-22.248575, -45.719977),
    (-22.233554, -45.711070),
    (-22.234289, -45.699882),
    (-22.268758, -45.714317),
    (-22.265084, -45.693242),
    (-22.252660, -45.687757),
    (-22.245780, -45.695335),
    (-22.250122, -45.688695),
]

def _cidade_polygon() -> Polygon:
    """Convex hull dos pontos extremos → polígono da cidade."""
    pts = np.array(_EXTREMOS)
    hull = ConvexHull(pts)
    verts = pts[hull.vertices]          # (lat, lon)
    coords = [(float(lon), float(lat)) for lat, lon in verts]
    coords.append(coords[0])            # fechar
    return Polygon(coords)

CIDADE_POLY: Polygon = _cidade_polygon()


# ── Cache CSV ─────────────────────────────────────────────────────────────────

CSV_PATH = Path("postes_srs.csv")
_df_postes: pd.DataFrame | None = None

def get_postes() -> pd.DataFrame:
    global _df_postes
    if _df_postes is None:
        if not CSV_PATH.exists():
            raise HTTPException(status_code=500, detail=f"{CSV_PATH} não encontrado.")
        _df_postes = pd.read_csv(CSV_PATH)
    return _df_postes


# ── Helpers geometria (bairros OSM) ──────────────────────────────────────────

def eq_ponto(a, b, tol=1e-7):
    return abs(a[0]-b[0]) < tol and abs(a[1]-b[1]) < tol

def fechar_anel(ring):
    if not eq_ponto(ring[0], ring[-1]):
        ring.append(ring[0])
    return ring

def unir_aneis(rings):
    if len(rings) == 1:
        return rings
    remaining = [list(r) for r in rings]
    merged = []
    while remaining:
        cur = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            for i, r in enumerate(remaining):
                if eq_ponto(cur[-1], r[0]):
                    cur = cur + r[1:]
                elif eq_ponto(cur[-1], r[-1]):
                    cur = cur + list(reversed(r))[1:]
                elif eq_ponto(cur[0], r[-1]):
                    cur = r + cur[1:]
                elif eq_ponto(cur[0], r[0]):
                    cur = list(reversed(r)) + cur[1:]
                else:
                    continue
                remaining.pop(i); changed = True; break
        merged.append(fechar_anel(cur))
    return merged

def overpass_para_geojson(data):
    features = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        nome = tags.get("name") or tags.get("name:pt")
        if not nome:
            continue
        if el["type"] == "way":
            geom = el.get("geometry", [])
            if len(geom) < 3:
                continue
            ring = [[p["lon"], p["lat"]] for p in geom]
            ring = fechar_anel(ring)
            if len(ring) < 4:
                continue
            features.append({"type": "Feature",
                "properties": {"name": nome, "colorIdx": len(features)},
                "geometry": {"type": "Polygon", "coordinates": [ring]}})
        elif el["type"] == "relation":
            outer_rings = []
            for m in el.get("members", []):
                if m.get("type") != "way" or m.get("role") not in ("outer", ""):
                    continue
                geom = m.get("geometry", [])
                if len(geom) < 3:
                    continue
                ring = [[p["lon"], p["lat"]] for p in geom]
                ring = fechar_anel(ring)
                if len(ring) >= 4:
                    outer_rings.append(ring)
            if not outer_rings:
                continue
            merged = unir_aneis(outer_rings)
            geometry = ({"type": "Polygon", "coordinates": merged}
                if len(merged) == 1
                else {"type": "MultiPolygon", "coordinates": [[r] for r in merged]})
            features.append({"type": "Feature",
                "properties": {"name": nome, "colorIdx": len(features)},
                "geometry": geometry})
    return {"type": "FeatureCollection", "features": features}


# ── Converter shapely → coordenadas GeoJSON ───────────────────────────────────

def shapely_para_coords(geom):
    """Converte Polygon ou MultiPolygon shapely para coordenadas GeoJSON."""
    if isinstance(geom, Polygon):
        return "Polygon", [[[x, y] for x, y in geom.exterior.coords]]
    elif isinstance(geom, MultiPolygon):
        coords = [[[x, y] for x, y in part.exterior.coords] for part in geom.geoms]
        return "MultiPolygon", [[c] for c in coords]
    return None, None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/geocode")
async def geocode(cidade: str = Query(...)):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": cidade, "format": "json", "limit": 1, "addressdetails": 1}
    headers = {"Accept-Language": "pt-BR", "User-Agent": "LumiGuard/1.0"}
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, params=params, headers=headers)
    results = resp.json()
    if not results:
        raise HTTPException(status_code=404, detail="Cidade não encontrada.")
    r = results[0]
    return {"lat": float(r["lat"]), "lon": float(r["lon"]),
            "display_name": r["display_name"],
            "osm_id": r["osm_id"], "osm_type": r["osm_type"]}


@app.get("/api/bairros")
async def bairros(osm_id: int = Query(...), osm_type: str = Query("relation")):
    tipo_map = {"relation": "r", "way": "w", "node": "n"}
    prefixo = tipo_map.get(osm_type, "r")
    query = f"""
[out:json][timeout:45];
{prefixo}({osm_id});
map_to_area->.cidade;
(
  relation["place"~"neighbourhood|suburb|quarter"](area.cidade);
  way["place"~"neighbourhood|suburb|quarter"](area.cidade);
);
out geom;"""
    async with httpx.AsyncClient(timeout=50) as client:
        resp = await client.post("https://overpass-api.de/api/interpreter", data={"data": query})
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Erro na Overpass API.")
    return overpass_para_geojson(resp.json())


# Cache dos labels do último K-Means rodado
_ultimo_estado: dict = {}   # {"k": int, "labels": np.ndarray}


@app.get("/api/clusters")
async def clusters(k: int = Query(15, ge=2, le=50)):
    """
    K-Means nos postes → convex hull de cada cluster → clip com polígono da cidade.
    Garante que nenhum cluster extrapola os limites da cidade.
    """
    df = get_postes()
    coords = df[["lat", "lon"]].values

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(coords)
    _ultimo_estado["k"] = k
    _ultimo_estado["labels"] = labels

    features = []
    for cluster_id in range(k):
        mask = labels == cluster_id
        pts = coords[mask]
        if len(pts) < 3:
            continue

        centroide = pts.mean(axis=0)

        # Convex hull do cluster
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            # shapely usa (lon, lat)
            hull_coords = [(float(p[1]), float(p[0])) for p in hull_pts]
            hull_poly = Polygon(hull_coords)
        except Exception:
            continue

        # ── Clip: intersetar com o polígono da cidade ─────────────────────────
        try:
            clipped = hull_poly.intersection(CIDADE_POLY)
        except Exception:
            clipped = hull_poly  # fallback sem clip

        if clipped.is_empty:
            continue

        tipo, coords_geo = shapely_para_coords(clipped)
        if tipo is None:
            continue

        features.append({
            "type": "Feature",
            "properties": {
                "cluster_id": cluster_id,
                "colorIdx": cluster_id,
                "total_postes": int(mask.sum()),
                "centroide_lat": round(float(centroide[0]), 6),
                "centroide_lon": round(float(centroide[1]), 6),
            },
            "geometry": {"type": tipo, "coordinates": coords_geo},
        })

    features.sort(key=lambda f: f["properties"]["total_postes"], reverse=True)
    return {"type": "FeatureCollection", "features": features}

@app.get("/api/postes/{cluster_id}")
async def postes_cluster(cluster_id: int, k: int = Query(15, ge=2, le=50)):
    """Retorna os postes (lat/lon) de um cluster específico como GeoJSON."""
    df = get_postes()

    # Reusar labels do cache se o k for o mesmo, senão rodar de novo
    if _ultimo_estado.get("k") != k:
        coords = df[["lat", "lon"]].values
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(coords)
        _ultimo_estado["k"] = k
        _ultimo_estado["labels"] = labels
    else:
        labels = _ultimo_estado["labels"]

    mask = labels == cluster_id
    subset = df[mask]

    features = [
        {
            "type": "Feature",
            "properties": {"bairro": row.bairro},
            "geometry": {"type": "Point", "coordinates": [row.lon, row.lat]},
        }
        for row in subset.itertuples()
    ]

    return {"type": "FeatureCollection", "features": features}
