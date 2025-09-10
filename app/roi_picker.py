# roi_picker.py
from flask import Blueprint, jsonify, request, render_template_string, Response

ROI_PICKER_HTML = """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>ROI Picker</title>
  <style>
    body{background:#0b1720;color:#d1fae5;font-family:system-ui,sans-serif}
    .wrap{max-width:1280px;margin:20px auto;padding:0 12px}
    #wrap{position:relative;display:inline-block}
    #main{display:block;max-width:100%;height:auto}
    #ov{position:absolute;left:0;top:0;pointer-events:none}
    .hint{margin-top:8px;color:#93c5fd}
    .links a{color:#86efac;text-decoration:none;margin-right:10px}
    .kbd{padding:2px 6px;border:1px solid #6b7280;border-radius:4px;background:#111827}
  </style>
</head>
<body>
  <div class="wrap">
    <h3>ROI 수동 설정</h3>
    <div id="wrap">
      <img id="main" src="/camera/main" alt="main stream">
      <canvas id="ov"></canvas>
    </div>
    <div class="hint" id="h">
      ▶ <span class="kbd">1~4</span> ZONE 선택 → 프레임에서 <b>4점</b> 클릭 → <span class="kbd">Enter</span> 적용, <span class="kbd">Esc</span> 취소,
      <span class="kbd">B</span> 원래 화면으로 돌아가기
    </div>
    <div class="links">
      <a href="/camera/1" target="_blank">/camera/1</a>
      <a href="/camera/2" target="_blank">/camera/2</a>
      <a href="/camera/3" target="_blank">/camera/3</a>
      <a href="/camera/4" target="_blank">/camera/4</a>
      <a href="#" id="reset">ROI 초기값으로</a>
    </div>
  </div>
  <script>
  (()=>{
    const img=document.getElementById('main');
    const cvs=document.getElementById('ov'); const ctx=cvs.getContext('2d');
    const hint=document.getElementById('h'); const reset=document.getElementById('reset');
    let frameW=0, frameH=0, sel=null, pts=[], lastW=0, lastH=0;

    async function fs(){
      try{const r=await fetch('/api/frame_size'); const j=await r.json();
        frameW=j.width||0; frameH=j.height||0;}catch(e){}
      layout();
    }
    setInterval(()=>{ if(!frameW||!frameH) fs(); }, 1500); fs();

    function layout(){
      const r=img.getBoundingClientRect(); const w=Math.round(r.width), h=Math.round(r.height);
      if(w>0&&h>0&&(w!==lastW||h!==lastH)){ cvs.width=w; cvs.height=h; cvs.style.width=w+'px'; cvs.style.height=h+'px';
        lastW=w; lastH=h; draw(); }
    }
    window.addEventListener('resize', layout); new ResizeObserver(layout).observe(img);

    function help(){ const z= sel?('ZONE '+sel):'미선택'; hint.innerHTML=
      '▶ <span class="kbd">1~4</span> ZONE 선택: <b>'+z+'</b> | 클릭 <b>'+pts.length+'/4</b> | <span class="kbd">Enter</span> 적용, <span class="kbd">Esc</span> 취소, <span class="kbd">B</span> 돌아가기'; }

    function draw(){
      ctx.clearRect(0,0,cvs.width,cvs.height);
      if(pts.length){
        ctx.lineWidth=2; ctx.strokeStyle='#22d3ee'; ctx.beginPath();
        for(let i=0;i<pts.length;i++){ const p=pts[i]; if(i===0) ctx.moveTo(p.dx,p.dy); else ctx.lineTo(p.dx,p.dy); }
        ctx.stroke();
        for(const p of pts){ ctx.beginPath(); ctx.arc(p.dx,p.dy,5,0,Math.PI*2); ctx.fillStyle='#22d3ee'; ctx.fill(); }
        if(pts.length>=2){
          const xs=pts.map(p=>p.dx), ys=pts.map(p=>p.dy);
          const x1=Math.min(...xs), y1=Math.min(...ys), x2=Math.max(...xs), y2=Math.max(...ys);
          ctx.strokeStyle='#fbbf24'; ctx.lineWidth=2; ctx.strokeRect(x1,y1,x2-x1,y2-y1);
        }
      }
      help();
    }

    function toFrame(e){
      const r=img.getBoundingClientRect(); const dx=e.clientX-r.left, dy=e.clientY-r.top;
      const sx=(frameW&&r.width)? frameW/r.width:1; const sy=(frameH&&r.height)? frameH/r.height:1;
      return {x:Math.round(dx*sx), y:Math.round(dy*sy), dx, dy};
    }

    img.addEventListener('click',(e)=>{
      if(!sel) return; const p=toFrame(e); pts.push(p); if(pts.length>4) pts=pts.slice(-4); draw();
    });

    window.addEventListener('keydown', async (e)=>{
      if(e.key>='1'&&e.key<='4'){ sel=parseInt(e.key,10); pts=[]; draw(); return; }
      if(e.key==='Escape'){ pts=[]; draw(); return; }
      if(e.key==='b' || e.key==='B'){
        try{ if(window.opener) window.opener.focus(); }catch(_){}
        window.close(); // 팝업이면 닫힘, 탭이면 무시됨 → 아래로 폴백
        if(!document.hasFocus()){ location.href='/'; } // 탭일 때는 메인으로 이동
        return;
      }
      if(e.key==='Enter'){
        if(!sel) return;
        if(pts.length!==4){ alert('정확히 4점을 찍어주세요.'); return; }
        const body={points: pts.map(p=>({x:p.x,y:p.y}))};
        try{
          const r=await fetch('/api/roi/'+sel,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
          const j=await r.json(); if(!j.ok) alert('적용 실패: '+(j.error||'unknown'));
          pts=[]; draw();
        }catch(err){ alert('네트워크 오류'); }
      }
    });

    reset.addEventListener('click', async (e)=>{ e.preventDefault(); try{ await fetch('/api/roi/reset',{method:'POST'}); }catch(_){ } });

    layout(); help();
  })();
  </script>
</body>
</html>
"""

# 메인 페이지에서 'r'을 누르면 /roi 팝업을 띄우는 아주 작은 스크립트
HOTKEYS_JS = """
(()=> {
  function openPicker(){
    try {
      const w = window.open('/roi','roi','width=1280,height=820,noopener,noreferrer');
      if (!w) { alert('팝업이 차단되었습니다. 브라우저 팝업 허용을 켜주세요.'); }
    } catch(e) {}
  }
  window.addEventListener('keydown', (e)=>{
    if (e.repeat) return;
    if (e.key === 'r' || e.key === 'R') { openPicker(); }
  });
})();
"""

def create_roi_blueprint(get_frame_size, get_all_zones, set_zone_bbox, reset_all):
    """
    외부(app.py)에서 내부 상태 접근 함수를 주입받아 /roi 및 /api/* 라우트를 제공
    """
    bp = Blueprint("roi", __name__)

    @bp.get("/roi")
    def roi_page():
        return render_template_string(ROI_PICKER_HTML)

    @bp.get("/roi/hotkeys.js")
    def hotkeys_js():
        return Response(HOTKEYS_JS, mimetype="application/javascript")

    @bp.get("/api/frame_size")
    def api_frame_size():
        W, H = get_frame_size()
        if not W or not H:
            return jsonify({"width": 0, "height": 0})
        return jsonify({"width": int(W), "height": int(H)})

    @bp.get("/api/roi")
    def api_get_roi():
        zones = get_all_zones()
        return jsonify({str(k): list(v) for k, v in zones.items()})

    @bp.post("/api/roi/<int:zid>")
    def api_set_roi(zid: int):
        data = request.get_json(silent=True) or {}
        pts = data.get("points", [])
        if not isinstance(pts, list) or len(pts) != 4:
            return jsonify({"ok": False, "error": "exactly 4 points required"}), 400

        W, H = get_frame_size()
        if not W or not H:
            return jsonify({"ok": False, "error": "frame size unknown yet"}), 409

        xs, ys = [], []
        for p in pts:
            try:
                x = float(p["x"]); y = float(p["y"])
            except Exception:
                return jsonify({"ok": False, "error": "invalid point format"}), 400
            x = max(0.0, min(W - 1.0, x))
            y = max(0.0, min(H - 1.0, y))
            xs.append(x); ys.append(y)

        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        MIN_W, MIN_H = 20, 20
        if (x2 - x1) < MIN_W: x2 = min(W - 1, x1 + MIN_W)
        if (y2 - y1) < MIN_H: y2 = min(H - 1, y1 + MIN_H)

        set_zone_bbox(zid, (x1, y1, x2, y2))
        return jsonify({"ok": True, "zone": zid, "bbox": [x1, y1, x2, y2]})

    @bp.post("/api/roi/reset")
    def api_reset_roi():
        reset_all()
        return jsonify({"ok": True})

    return bp
