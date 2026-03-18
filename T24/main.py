import httpx
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

# Serve o front em /static/ (coloque index.html na pasta "static/")
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


# ── Helpers de geometria ──────────────────────────────────────────────────────

def eq_ponto(a: list, b: list, tol: float = 1e-7) -> bool:
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


def fechar_anel(ring: list) -> list:
    if not eq_ponto(ring[0], ring[-1]):
        ring.append(ring[0])
    return ring


def unir_aneis(rings: list[list]) -> list[list]:
    """Une anéis que compartilham extremos (head-to-tail) num único anel contínuo."""
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
                remaining.pop(i)
                changed = True
                break

        merged.append(fechar_anel(cur))

    return merged


def overpass_para_geojson(data: dict) -> dict:
    """Converte resposta Overpass (out geom) em GeoJSON FeatureCollection."""
    features = []

    for el in data.get("elements", []):
        tags = el.get("tags", {})
        nome = tags.get("name") or tags.get("name:pt")
        if not nome:
            continue

        # Way simples com geometria embutida
        if el["type"] == "way":
            geom = el.get("geometry", [])
            if len(geom) < 3:
                continue
            ring = [[p["lon"], p["lat"]] for p in geom]
            ring = fechar_anel(ring)
            if len(ring) < 4:
                continue
            features.append({
                "type": "Feature",
                "properties": {"name": nome, "colorIdx": len(features)},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            })

        # Relação com membros
        elif el["type"] == "relation":
            outer_rings = []
            for m in el.get("members", []):
                if m.get("type") != "way":
                    continue
                if m.get("role") not in ("outer", ""):
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
            geometry = (
                {"type": "Polygon", "coordinates": merged}
                if len(merged) == 1
                else {"type": "MultiPolygon", "coordinates": [[r] for r in merged]}
            )
            features.append({
                "type": "Feature",
                "properties": {"name": nome, "colorIdx": len(features)},
                "geometry": geometry,
            })

    return {"type": "FeatureCollection", "features": features}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/geocode")
async def geocode(cidade: str = Query(..., description="Nome da cidade")):
    """Geocodifica uma cidade via Nominatim e retorna lat/lon."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": cidade, "format": "json", "limit": 1}
    headers = {"Accept-Language": "pt-BR", "User-Agent": "LumiGuard/1.0"}

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, params=params, headers=headers)

    if resp.status_code != 200 or not resp.json():
        raise HTTPException(status_code=404, detail="Cidade não encontrada.")

    result = resp.json()[0]
    return {
        "lat": float(result["lat"]),
        "lon": float(result["lon"]),
        "display_name": result["display_name"],
    }


@app.get("/api/bairros")
async def bairros(
    lat: float = Query(...),
    lon: float = Query(...),
    raio_km: int = Query(15, ge=1, le=50, description="Raio de busca em km"),
):
    """Busca bairros via Overpass API e retorna GeoJSON pronto para o Leaflet."""
    raio_m = raio_km * 1000

    # Busca apenas elementos com place=neighbourhood/suburb/quarter
    # contidos dentro da área urbana da cidade (raio menor, 5km)
    # para evitar pegar municípios vizinhos ou áreas rurais enormes
    raio_bairro = min(raio_m, 5000)
    query = f"""
[out:json][timeout:30];
(
  relation["place"~"neighbourhood|suburb|quarter"](around:{raio_bairro},{lat},{lon});
  way["place"~"neighbourhood|suburb|quarter"](around:{raio_bairro},{lat},{lon});
  node["place"~"neighbourhood|suburb|quarter"](around:{raio_bairro},{lat},{lon});
);
out geom;"""

    async with httpx.AsyncClient(timeout=35) as client:
        resp = await client.post(
            "https://overpass-api.de/api/interpreter",
            data={"data": query},
        )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail="Erro na Overpass API.")

    geojson = overpass_para_geojson(resp.json())
    return geojson
