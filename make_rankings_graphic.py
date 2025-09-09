
# make_rankings_graphic.py
import argparse
import glob
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# --- Pillow resampling compatibility (covers all versions) ---
if hasattr(Image, "Resampling"):       # Pillow ≥ 10
    RESAMPLE = Image.Resampling.LANCZOS
elif hasattr(Image, "LANCZOS"):        # Pillow 7–9
    RESAMPLE = Image.LANCZOS
elif hasattr(Image, "ANTIALIAS"):      # Older Pillow
    RESAMPLE = Image.ANTIALIAS
else:
    RESAMPLE = Image.BICUBIC           # Last resort

# ----------- Base config -----------
PORTRAIT_W, PORTRAIT_H = 1080, 1350
SQUARE_W, SQUARE_H = 1080, 1080
MARGIN = 48
PAD = 18

BG = (10, 10, 12)
PANEL_BG = (18, 18, 22)
ACCENT = (247, 191, 39)
TEXT = (240, 240, 245)
SUBTEXT = (180, 182, 190)
BAR = (255, 215, 64)

def load_font(size, bold=False):
    candidates = [
        ("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        ("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

FONT_H1 = load_font(64, bold=True)
FONT_H2 = load_font(42, bold=True)
FONT_H3 = load_font(30, bold=True)
FONT_BODY = load_font(26, bold=False)
FONT_SMALL = load_font(22, bold=False)

def draw_text(draw, xy, text, font, fill=TEXT, anchor="la"):
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)

def rounded_rect(draw, box, radius, fill):
    draw.rounded_rectangle(box, radius=radius, fill=fill)

def load_image_fit(path, max_w, max_h):
    p = Path(path) if path else None
    if p and p.exists():
        img = Image.open(p).convert("RGBA")
        w, h = img.size
        scale = min(max_w / w, max_h / h) if max_w and max_h else 1.0
        if scale < 1:
            img = img.resize((int(w*scale), int(h*scale)), RESAMPLE)
        return img
    return None

def load_team_logo(team_dir, team_code, max_w, max_h):
    if not team_dir:
        return None
    p = Path(team_dir) / f"{team_code}.png"
    if p.exists():
        return load_image_fit(p, max_w, max_h)
    return None

def fit_prob(x):
    try:
        return float(max(0, min(1, float(x))))
    except Exception:
        return 0.0

def format_stat(val, digits=1):
    if pd.isna(val):
        return "-"
    try:
        return f"{float(val):.{digits}f}"
    except Exception:
        return str(val)

def split_leagues(df, prob_col):
    al = df[df["lgID"] == "AL"].sort_values(prob_col, ascending=False).head(5).reset_index(drop=True)
    nl = df[df["lgID"] == "NL"].sort_values(prob_col, ascending=False).head(5).reset_index(drop=True)
    return al, nl

# ---------- NEW: resolve latest file from pattern ----------
def resolve_latest(path_or_pattern: str) -> str:
    """If a wildcard is present, return the newest match by mtime; else return as-is (after checking exists)."""
    if any(ch in path_or_pattern for ch in "*?[]"):
        matches = glob.glob(path_or_pattern)
        if not matches:
            raise FileNotFoundError(f"No files match pattern: {path_or_pattern}")
        # pick by modification time
        latest = max(matches, key=os.path.getmtime)
        print(f"→ Using latest match for '{path_or_pattern}': {latest}")
        return latest
    # no wildcard: just validate
    p = Path(path_or_pattern)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path_or_pattern}")
    return str(p)

# --- helper: truncate text to fit pixel width ---
def truncate_text(draw, text, font, max_px):
    if draw.textlength(text, font=font) <= max_px:
        return text
    ell = "…"
    if draw.textlength(ell, font=font) > max_px:
        return ""  # too tiny; return empty
    # binary search cut
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        candidate = text[:mid] + ell
        if draw.textlength(candidate, font=font) <= max_px:
            lo = mid + 1
        else:
            hi = mid
    return text[:lo-1] + ell

# ----------- Header with gradient (updated: avoid double-pasting gradient as logo) -----------
def draw_header(canvas, title, subtitle=None, logo_path=None, gradient_path=None, height=150):
    draw = ImageDraw.Draw(canvas)
    y = MARGIN
    paste_logo = bool(logo_path) and (not gradient_path or str(logo_path) != str(gradient_path))

    if gradient_path:
        grad = Image.open(gradient_path).convert("RGBA")
        grad = grad.resize((canvas.width - 2*MARGIN, height), RESAMPLE)
        canvas.paste(grad, (MARGIN, y))
        x_text = MARGIN + 20
        if paste_logo:
            logo = load_image_fit(logo_path, max_w=180, max_h=height-20)
            if logo:
                canvas.paste(logo, (MARGIN + 20, y + 10), logo)
                x_text = MARGIN + 20 + logo.width + 16
        draw_text(draw, (x_text, y + 16), title, FONT_H1, fill=(10,10,10), anchor="la")
        if subtitle:
            draw_text(draw, (x_text, y + 16 + 64), subtitle, FONT_BODY, fill=(20,20,20), anchor="la")
    else:
        rounded_rect(draw, (MARGIN, y, canvas.width - MARGIN, y + height), 24, PANEL_BG)
        x_text = MARGIN + 24
        if paste_logo:
            logo = load_image_fit(logo_path, max_w=180, max_h=height-20)
            if logo:
                canvas.paste(logo, (x_text, y + 10), logo)
                x_text += logo.width + 16
        draw_text(draw, (x_text, y + 16), title, FONT_H1)
        if subtitle:
            draw_text(draw, (x_text, y + 16 + 64), subtitle, FONT_BODY, fill=SUBTEXT)

# ----------- Leaderboard panel (updated layout & truncation) -----------
def draw_leaderboard(canvas, x, y, w, h, title, df, prob_col, team_logo_dir=None, mode="MVP"):
    draw = ImageDraw.Draw(canvas)
    rounded_rect(draw, (x, y, x + w, y + h), 24, PANEL_BG)

    # Header
    draw_text(draw, (x + 20, y + 18), title, FONT_H2, fill=ACCENT)

    # Column anchors (more breathing room)
    col_y = y + 72
    draw_text(draw, (x + 20, col_y), "Rk", FONT_SMALL, fill=SUBTEXT)

    # reserve ~44px for rank + padding + optional team logo
    name_x = x + 20 + 36 + 44
    team_x = x + w - 380
    prob_x = x + w - 250
    stat1_x = x + w - 150
    stat2_x = x + w - 90

    draw_text(draw, (name_x, col_y), "Player", FONT_SMALL, fill=SUBTEXT)
    draw_text(draw, (team_x, col_y), "Team", FONT_SMALL, fill=SUBTEXT)
    draw_text(draw, (prob_x, col_y), "Prob", FONT_SMALL, fill=SUBTEXT)
    if mode == "MVP":
        draw_text(draw, (stat1_x, col_y), "WAR", FONT_SMALL, fill=SUBTEXT)
        draw_text(draw, (stat2_x, col_y), "wRC+", FONT_SMALL, fill=SUBTEXT)
    else:
        draw_text(draw, (stat1_x, col_y), "pWAR", FONT_SMALL, fill=SUBTEXT)
        draw_text(draw, (stat2_x, col_y), "FIP",  FONT_SMALL, fill=SUBTEXT)

    # Row sizing (minimum height to avoid overlap)
    row_h = max((h - 110) // 5, 64)
    y0 = col_y + 16

    for i in range(len(df)):
        r = df.iloc[i]
        ry = y0 + i * row_h + 6
        rounded_rect(draw, (x + 12, ry, x + w - 12, ry + row_h - 12), 18, (25,25,30))

        # Rank
        draw_text(draw, (x + 24, ry + row_h//2), f"{i+1}", FONT_H3, anchor="lm")

        # Team logo (optional)
        logo_px = 36
        team_code = str(r.get("Team", "-"))
        team_logo = load_team_logo(team_logo_dir, team_code, max_w=logo_px, max_h=logo_px)
        px_after_rank = x + 24 + 32  # rank area width
        name_start = px_after_rank + 12
        if team_logo:
            canvas.paste(team_logo, (px_after_rank + 12, ry + (row_h - logo_px)//2), team_logo)
            name_start += logo_px + 12

        # Player name (truncate)
        player_name = str(r.get("Name", r.get("playerID", "Unknown")))
        max_name_px = team_x - 20 - name_start
        player_name = truncate_text(draw, player_name, FONT_BODY, max_name_px)
        draw_text(draw, (name_start, ry + row_h//2), player_name, FONT_BODY, anchor="lm")

        # Team code (truncate, small)
        max_team_px = prob_x - 24 - team_x
        team_disp = truncate_text(draw, team_code, FONT_BODY, max_team_px)
        draw_text(draw, (team_x, ry + row_h//2), team_disp, FONT_BODY, fill=SUBTEXT, anchor="lm")

        # Probability bar (right side)
        p = fit_prob(r.get(prob_col, 0))
        bar_w, bar_h = 120, 20
        bx = prob_x
        by = ry + (row_h - bar_h)//2
        rounded_rect(draw, (bx, by, bx + bar_w, by + bar_h), 10, (40,40,48))
        rounded_rect(draw, (bx, by, bx + int(bar_w * p), by + bar_h), 10, BAR)
        pct_text = f"{p*100:,.0f}%"
        draw_text(draw, (bx + bar_w + 6, by + bar_h//2), pct_text, FONT_SMALL, anchor="lm")

        # Stats (right aligned numbers)
        if mode == "MVP":
            war = r.get("bat_WAR_fg", np.nan)
            wrc = r.get("bat_wRC_plus", np.nan)
            draw_text(draw, (stat1_x + 40, ry + row_h//2), format_stat(war, 1), FONT_BODY, anchor="rm")
            draw_text(draw, (stat2_x + 40, ry + row_h//2), format_stat(wrc, 0), FONT_BODY, anchor="rm")
        else:
            pwar = r.get("pit_WAR_fg", np.nan)
            fip  = r.get("pit_FIP", np.nan)
            draw_text(draw, (stat1_x + 40, ry + row_h//2), format_stat(pwar, 1), FONT_BODY, anchor="rm")
            draw_text(draw, (stat2_x + 40, ry + row_h//2), format_stat(fip, 2),  FONT_BODY, anchor="rm")


# ----------- Portrait 1080×1350 -----------
def render_portrait(year, al_mvp, nl_mvp, al_cy, nl_cy, out_png, logo=None, gradient=None, team_logos=None):
    img = Image.new("RGB", (PORTRAIT_W, PORTRAIT_H), BG)
    draw_header(img, f"{year} Award Races", "Top 5 • Stinger Collectibles", logo_path=logo, gradient_path=gradient, height=150)

    grid_x = MARGIN
    grid_y = MARGIN + 160
    grid_w = PORTRAIT_W - 2*MARGIN
    grid_h = PORTRAIT_H - grid_y - MARGIN
    col_w = (grid_w - PAD) // 2
    row_h = (grid_h - PAD) // 2

    draw_leaderboard(img, grid_x, grid_y, col_w, row_h, "AL MVP — Top 5", al_mvp, "MVP_prob", team_logo_dir=team_logos, mode="MVP")
    draw_leaderboard(img, grid_x + col_w + PAD, grid_y, col_w, row_h, "NL MVP — Top 5", nl_mvp, "MVP_prob", team_logo_dir=team_logos, mode="MVP")
    draw_leaderboard(img, grid_x, grid_y + row_h + PAD, col_w, row_h, "AL Cy Young — Top 5", al_cy, "CY_prob", team_logo_dir=team_logos, mode="CY")
    draw_leaderboard(img, grid_x + col_w + PAD, grid_y + row_h + PAD, col_w, row_h, "NL Cy Young — Top 5", nl_cy, "CY_prob", team_logo_dir=team_logos, mode="CY")

    d = ImageDraw.Draw(img)
    foot = "Inputs: FanGraphs WAR, wRC+, OPS, FIP, team Win%, plus counting stats • auto-generated"
    d.text((PORTRAIT_W//2, PORTRAIT_H - MARGIN//2), foot, font=FONT_SMALL, fill=SUBTEXT, anchor="ms")

    outp = Path(out_png)
    outp.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(outp), format="PNG")
    print(f"✅ Saved portrait → {outp.resolve()}")

# ----------- Carousel 1080×1080 -----------
def square_header(canvas, title, subtitle=None, gradient=None, logo=None):
    draw = ImageDraw.Draw(canvas)
    if gradient:
        grad = Image.open(gradient).convert("RGBA").resize((SQUARE_W - 2*MARGIN, 260), RESAMPLE)
        canvas.paste(grad, (MARGIN, MARGIN))
        x = MARGIN + 24
        if logo:
            lg = load_image_fit(logo, max_w=200, max_h=140)
            if lg:
                canvas.paste(lg, (x, MARGIN + 20), lg)
                x += lg.width + 16
        draw_text(draw, (x, MARGIN + 24), title, FONT_H1, fill=(10,10,10))
        if subtitle:
            draw_text(draw, (x, MARGIN + 24 + 64), subtitle, FONT_BODY, fill=(20,20,20))
    else:
        rounded_rect(draw, (MARGIN, MARGIN, SQUARE_W - MARGIN, MARGIN + 220), 24, PANEL_BG)
        x = MARGIN + 24
        if logo:
            lg = load_image_fit(logo, max_w=200, max_h=140)
            if lg:
                canvas.paste(lg, (x, MARGIN + 20), lg)
                x += lg.width + 16
        draw_text(draw, (x, MARGIN + 24), title, FONT_H1)
        if subtitle:
            draw_text(draw, (x, MARGIN + 24 + 64), subtitle, FONT_BODY, fill=SUBTEXT)

def render_square_panel(title_left, df_left, prob_left, title_right, df_right, prob_right, out_path, gradient=None, logo=None, team_logos=None, mode_left="MVP", mode_right="MVP"):
    img = Image.new("RGB", (SQUARE_W, SQUARE_H), BG)
    square_header(img, title="Stinger Award Races", subtitle=title_left.split("—")[0].strip()+" • "+title_right.split("—")[0].strip(), gradient=gradient, logo=logo)
    grid_x = MARGIN
    grid_y = MARGIN + 260 + 12
    grid_w = SQUARE_W - 2*MARGIN
    col_w = (grid_w - PAD) // 2
    panel_h = SQUARE_H - grid_y - MARGIN

    draw_leaderboard(img, grid_x, grid_y, col_w, panel_h, title_left, df_left, prob_left, team_logo_dir=team_logos, mode=mode_left)
    draw_leaderboard(img, grid_x + col_w + PAD, grid_y, col_w, panel_h, title_right, df_right, prob_right, team_logo_dir=team_logos, mode=mode_right)

    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(outp), format="PNG")
    print(f"✅ Saved square → {outp.resolve()}")

def render_square_cover(year, out_path, gradient=None, logo=None):
    img = Image.new("RGB", (SQUARE_W, SQUARE_H), BG)
    square_header(img, title=f"{year} Award Races", subtitle="Top 5 • Stinger Collectibles", gradient=gradient, logo=logo)
    d = ImageDraw.Draw(img)
    msg = "MVP & Cy Young • AL & NL\nUpdated automatically"
    draw_text(d, (SQUARE_W//2, SQUARE_H//2 + 80), msg, FONT_H2, fill=SUBTEXT, anchor="ms")
    outp = Path(out_path); outp.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(outp), format="PNG")
    print(f"✅ Saved cover → {outp.resolve()}")

def render_square_outro(out_path, gradient=None, logo=None):
    img = Image.new("RGB", (SQUARE_W, SQUARE_H), BG)
    square_header(img, title="Follow @StingerCollectibles", subtitle="Weekly award race updates", gradient=gradient, logo=logo)
    d = ImageDraw.Draw(img)
    draw_text(d, (SQUARE_W//2, SQUARE_H//2 + 60), "Predictions based on ML models trained on 1980–present data", FONT_BODY, fill=SUBTEXT, anchor="ms")
    outp = Path(out_path); outp.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(outp), format="PNG")
    print(f"✅ Saved outro → {outp.resolve()}")

# ----------- Build from CSVs -----------
def load_and_split(mvp_csv_arg, cy_csv_arg):
    mvp_csv = resolve_latest(mvp_csv_arg)
    cy_csv  = resolve_latest(cy_csv_arg)
    mvp = pd.read_csv(mvp_csv)
    cy  = pd.read_csv(cy_csv)
    al_mvp, nl_mvp = split_leagues(mvp, "MVP_prob")
    al_cy,  nl_cy  = split_leagues(cy,  "CY_prob")
    return al_mvp, nl_mvp, al_cy, nl_cy

# ----------- CLI -----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--mvp_csv", type=str, required=True, help="File or glob pattern for MVP CSV (e.g., 'predictions/2025/top5_mvp_2025_*.csv')")
    ap.add_argument("--cy_csv", type=str, required=True,  help="File or glob pattern for CY CSV (e.g., 'predictions/2025/top5_cy_2025_*.csv')")
    ap.add_argument("--out", type=str, help="Portrait PNG output path (optional if exporting carousel)")
    ap.add_argument("--export_carousel", type=str, help="Folder to write 4-square carousel (cover, MVP, CY, outro)")
    ap.add_argument("--logo", type=str, default=None)
    ap.add_argument("--gradient", type=str, default=None)
    ap.add_argument("--team_logos", type=str, default=None)
    args = ap.parse_args()

    al_mvp, nl_mvp, al_cy, nl_cy = load_and_split(args.mvp_csv, args.cy_csv)

    if args.out:
        render_portrait(args.year, al_mvp, nl_mvp, al_cy, nl_cy, args.out,
                        logo=args.logo, gradient=args.gradient, team_logos=args.team_logos)

    if args.export_carousel:
        outdir = Path(args.export_carousel); outdir.mkdir(parents=True, exist_ok=True)
        render_square_cover(args.year, outdir / "01_cover.png", gradient=args.gradient, logo=args.logo)
        render_square_panel("AL MVP — Top 5", al_mvp, "MVP_prob",
                            "NL MVP — Top 5", nl_mvp, "MVP_prob",
                            outdir / "02_mvp.png", gradient=args.gradient, logo=args.logo, team_logos=args.team_logos,
                            mode_left="MVP", mode_right="MVP")
        render_square_panel("AL Cy Young — Top 5", al_cy, "CY_prob",
                            "NL Cy Young — Top 5", nl_cy, "CY_prob",
                            outdir / "03_cy.png", gradient=args.gradient, logo=args.logo, team_logos=args.team_logos,
                            mode_left="CY", mode_right="CY")
        render_square_outro(outdir / "04_outro.png", gradient=args.gradient, logo=args.logo)
