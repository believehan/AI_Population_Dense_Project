# geo_fix.py
# pip install geopy geographiclib

from typing import Tuple, Optional, List
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geographiclib.geodesic import Geodesic

Point = Tuple[float, float]

def distance_m(a: Point, b: Point) -> float:
    g = Geodesic.WGS84.Inverse(a[0], a[1], b[0], b[1])
    return g["s12"]

def polygon_area_m2(points: List[Point]) -> float:
    poly = Geodesic.WGS84.PolygonArea()
    for lat, lon in points:
        poly.AddPoint(lat, lon)
    _, _, area = poly.Compute()
    return abs(area)

def geocode_strict(geocode, q: str, viewbox=None) -> Optional[Point]:
    """
    viewbox: (W, S, E, N)  또는 ((W,S),(E,N))
    """
    loc = geocode(
        q,
        addressdetails=False,
        language="en",
        country_codes="us",
        viewbox=viewbox,
        bounded=True,
        exactly_one=True
    )
    if not loc:
        return None
    return (loc.latitude, loc.longitude)

if __name__ == "__main__":
    # 1) 지오코더(연락처 포함 User-Agent 필수)
    ua = "geo-measure/1.0 (contact: your_email@example.com)"
    geolocator = Nominatim(user_agent=ua)
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)  # 예의상 1초 간격

    # 2) 맨해튼 대략적 범위(서,남,동,북)
    MANHATTAN_VIEWBOX = (-74.02, 40.70, -73.90, 40.92)

    # 3) 동일 장소에 대한 후보 질의(순서대로 시도)
    candidates = [
        "TKTS Times Square",
        "Father Duffy Square, New York",
        "Broadway & W 47th St, New York, NY",
        "George M. Cohan Statue, Times Square",
    ]

    got = None
    for q in candidates:
        got = geocode_strict(geocode, q, viewbox=MANHATTAN_VIEWBOX)
        print(f"try: {q} -> {got}")
        if got:
            break
    if not got:
        raise SystemExit("모든 후보 지오코딩 실패: 검색어를 바꾸거나 좌표를 직접 넣어 주세요.")

    # 4) 예시: TKTS 주변 두 지점 거리 혹은 교차로(4점) 면적 계산
    #    (여기서는 얻어진 got(예: Father Duffy Square)와, 인근 고정 좌표를 예시로 사용)
    father_duffy = got

    # 인근 'George M. Cohan' 동상 좌표(위키 문서에 기재)
    cohan = (40.7587583, -73.9851444)

    print("\n[예시] Father Duffy ↔ George M. Cohan 거리(m):",
          round(distance_m(father_duffy, cohan), 2))

    # 교차로 모서리 4점을 직접 지도에서 복사해 넣으면 면적(㎡) 계산 가능
    # corners = [(lat1,lon1), (lat2,lon2), (lat3,lon3), (lat4,lon4)]
    # print("교차로 면적(㎡):", round(polygon_area_m2(corners), 2))
