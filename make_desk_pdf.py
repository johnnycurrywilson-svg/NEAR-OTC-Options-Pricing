"""Clean 3-page NEAR trading-desk research note, with a frozen as-of timestamp."""
from __future__ import annotations
import math
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

W,H=A4
NAVY=HexColor('#152537');INK=HexColor('#273543');MUTED=HexColor('#596a77')
RULE=HexColor('#d7e0e6');ACCENT=HexColor('#3c657b');SOFT=HexColor('#f2f5f7')
M=46

def setup_fonts():
    import os
    regular='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    if os.path.isfile(regular):
        pdfmetrics.registerFont(TTFont('DS',regular))
        pdfmetrics.registerFont(TTFont('DS-Bold',bold))
    else:
        # Built-in PostScript fallback works on Windows without font downloads.
        from reportlab.pdfbase.pdfmetrics import Font
        pdfmetrics.registerFont(Font('DS','Helvetica','WinAnsiEncoding'))
        pdfmetrics.registerFont(Font('DS-Bold','Helvetica-Bold','WinAnsiEncoding'))
    # Required for <b> and <i> inside ReportLab Paragraphs on Windows.
    # Without this mapping the parser cannot resolve the custom 'DS' font family.
    pdfmetrics.registerFontFamily('DS', normal='DS', bold='DS-Bold', italic='DS', boldItalic='DS-Bold')


def ptext(c,txt,x,y,width,size=9.1,leading=14,color=INK,bold=False):
    sty=ParagraphStyle('p',fontName='DS-Bold' if bold else 'DS',fontSize=size,leading=leading,textColor=color,
                       spaceBefore=0,spaceAfter=0,allowWidows=0,allowOrphans=0)
    para=Paragraph(txt,sty); ww,hh=para.wrap(width,1000);para.drawOn(c,x,y-hh)
    return y-hh


def label(c,txt,y):
    c.setFont('DS-Bold',9.0);c.setFillColor(ACCENT);c.drawString(M,y,txt.upper())
    c.setStrokeColor(RULE);c.setLineWidth(.6);c.line(M,y-8,W-M,y-8)
    return y-23


def header(c,idx,asof):
    c.setFillColor(NAVY);c.setFont('DS-Bold',9)
    c.drawString(M,H-37,'NEAR / OTC OPTIONS RESEARCH')
    c.setFillColor(MUTED);c.setFont('DS',8)
    c.drawRightString(W-M,H-37,f'12 NOV 2026  |  SNAPSHOT {asof[11:16]} UTC')
    c.setStrokeColor(RULE);c.line(M,H-44,W-M,H-44)
    c.setStrokeColor(RULE);c.line(M,49,W-M,49)
    c.setFont('DS',7.5);c.drawString(M,35,'Illustrative two-way option pricing. Not an executable dealer quote.')
    c.drawRightString(W-M,35,f'{idx} / 3')


def title(c,t,sub,y):
    c.setFillColor(NAVY);c.setFont('DS-Bold',19);c.drawString(M,y,t)
    c.setFillColor(MUTED);c.setFont('DS',9.0);c.drawString(M,y-21,sub)
    return y-49


def simple_table(c,headers,rows,y,widths,rowh=24,fontsize=8.4,highlight=-1):
    x=M
    c.setFillColor(SOFT);c.rect(M,y-rowh,sum(widths),rowh,fill=1,stroke=0)
    for i,h in enumerate(headers):
        c.setFillColor(NAVY);c.setFont('DS-Bold',fontsize-0.5)
        c.drawString(x+7,y-rowh+8,str(h))
        x+=widths[i]
    y-=rowh
    for j,row in enumerate(rows):
        if j==highlight:
            c.setFillColor(HexColor('#e9f0f3'));c.rect(M,y-rowh,sum(widths),rowh,fill=1,stroke=0)
        c.setStrokeColor(RULE);c.line(M,y-rowh,M+sum(widths),y-rowh)
        x=M
        for i,val in enumerate(row):
            c.setFillColor(INK);c.setFont('DS-Bold' if j==highlight else 'DS',fontsize)
            c.drawString(x+7,y-rowh+8,str(val));x+=widths[i]
        y-=rowh
    return y


def soft_box(c,x,y,width,height):
    c.setFillColor(SOFT);c.roundRect(x,y,width,height,6,fill=1,stroke=0)


def money(x):return f'${x:.3f}'
def pct(x):return f'{100*x:.1f}%'


def create_pdf(d,filepath):
    setup_fonts();c=canvas.Canvas(str(filepath),pagesize=A4)
    c.setTitle('NEAR OTC Option Pricing | 12 November 2026')
    at=d['asof_utc'];r=d['ticket_quotes'];atm=r['ATM call'];spot=d['spot']
    fwd=pct(d['fixed_forward_rv']);atmiv=pct(d['fair_atm_iv']);upl=pct(d['peer_relative_adjustment'])
    # PAGE 1: input -> volatility models -> surface
    header(c,1,at)
    y=title(c,'A NEAR option pricing example','12 November 2026 | Python research model | neutral two-way quote',H-81)
    y=ptext(c,'This project uses <b>Python</b> to collect options IV from Derive and Deribit, calculate NEAR realised volatility from daily closes, and fit the shape of a NEAR volatility surface. We then price an option with Black-76 and add a bid/offer allowance for hedging and risk. The numbers below use the recorded 12 November market snapshot.',M,y,W-2*M,9.2,15)
    y-=18
    y=label(c,'Market inputs',y)
    w=(W-2*M)/4
    for i,(k,v) in enumerate([('NEAR spot',f'${spot:.2f}'),('30d realised',pct(d['rv_30'])),('30d RV forecast',fwd),('30d fair ATM IV',atmiv)]):
        xx=M+i*w
        c.setFillColor(MUTED);c.setFont('DS',8);c.drawString(xx+3,y,k)
        c.setFillColor(NAVY);c.setFont('DS-Bold',16);c.drawString(xx+3,y-22,v)
    y-=54
    y=label(c,'How we forecast volatility',y)
    y=ptext(c,'We tested five methods on the same NEAR price history. Historical accuracy is a useful check, but a winning backtest does not guarantee a sound live quote.',M,y,W-2*M,8.8,13)
    y-=10
    original=d['experimental_har']['har_rv_pct']
    alt=d.get('all_forecast_models',{})
    rows=[('Trailing RV30','Last 30 days of daily price moves',f"{d['rv_30']*100:.1f}%"),
          ('Fixed blend','20% / 45% / 35% of 7/30/90d variance',fwd),
          ('Optimised blend','Choose weights using matured past outcomes',f"{100*alt['Optimised blend']:.1f}%*"),
          ('EWMA','More weight on recent daily returns',f"{100*alt['EWMA']:.1f}%*"),
          ('HAR ridge','Regression of past RV; research challenger',f'{original:.1f}%*')]
    y=simple_table(c,['Method','Method in plain English','12 Nov forecast'],rows,y,[113,273,115],rowh=26,fontsize=8,highlight=1)
    y-=12
    y=ptext(c,'<b>Pricing choice:</b> use the fixed blend as our baseline, not HAR. HAR won a small, overlapping holdout but its live forecast was unusually low and its underprediction risk remains untested. The asterisked forecasts are diagnostics, not pricing inputs.',M,y,W-2*M,8.5,13)
    y-=18
    y=label(c,'How peer options determine the fair IV',y)
    y=ptext(c,f'The four qualifying assets are <b>HYPE, SOL, AVAX and ZEC</b> (TRX lacked a usable fit). We compare each asset\'s current 30-day option IV against its own trailing 30-day realised volatility. A quality-weighted robust average gives a <b>{upl} relative adjustment</b>. Applied to {fwd} forecast RV, it gives <b>{atmiv} fair ATM IV</b>. Their normalised smiles set the relative strike and expiry shape, not NEAR\'s absolute volatility.',M,y,W-2*M,9.0,14)
    y-=12
    y=ptext(c,'Source snapshot: Derive 359 option nodes, Deribit 498 option nodes and Kraken NEAR/USD, observed 12 November 2026 at 07:41 UTC. Source marks and theoretical quotes have not been verified for executable size.',M,y,W-2*M,7.6,11,color=MUTED)
    c.showPage()
    # PAGE 2: show quotes & spreads + scenarios
    header(c,2,at)
    y=title(c,'From fair IV to a two-way quote','Black-76 European USD cash-settled options | 30 days | zero carry and discounting',H-81)
    y=ptext(c,f'We assume a NEAR forward of <b>${spot:.2f}</b> (equal to spot), calculate the theoretical fair premium, then quote below and above that fair value. Both sides use the same volatility surface; there is <b>no built-in buying or selling bias</b>.',M,y,W-2*M,9.2,14)
    y-=22
    y=label(c,'Client RFQ examples',y)
    quotes=[('ATM call',r['ATM call']),('20% OTM call',r['20% OTM covered call']),('15% OTM put',r['15% OTM cash-secured put'])]
    rows=[(nm,f"${q['strike']:.2f}",f"{q['fair_iv_pct']:.1f}%",money(q['bid_usd']),money(q['fair_usd']),money(q['offer_usd'])) for nm,q in quotes]
    y=simple_table(c,['Option','Strike','Fair IV','Dealer bid','Fair mid','Dealer offer'],rows,y,[115,64,72,80,80,90],27,8.1,highlight=0)
    y-=14
    y=ptext(c,f'For the ATM call, our <b>{atmiv}</b> fair-volatility estimate becomes <b>{atm["bid_iv_pct"]:.1f}% bid / {atm["offer_iv_pct"]:.1f}% offer</b>. A dealer buys the option at the bid and sells it at the offer. The illustrative call midpoint is <b>{money(atm["fair_usd"])}</b> per NEAR.',M,y,W-2*M,8.9,14)
    y-=19
    y=label(c,'What goes into the spread',y)
    rows=[('Estimated hedge execution / rebalancing',f"{atm['execution_vol_points_per_side']:.2f} vol pts / side"),
          ('Model uncertainty reserve','1.00 vol pt / side'),('Jump-risk reserve','1.00 vol pt / side'),
          ('FULL ATM SPREAD',f"{atm['total_spread_vol_points']:.2f} vol pts")]
    y=simple_table(c,['Source of width','Model allowance'],rows,y,[325,176],24,8.1,highlight=3)
    y-=12
    y=ptext(c,'The base case assumes <b>30 rehedges in 30 days</b> and 15 basis points per hedge (5 bps fees, 5 bps half-spread, 5 bps slippage). We estimate the option\'s delta and gamma turnover, convert that cost into dollars, then divide by vega to express it in volatility points. These are <b>assumptions</b>, not measured lifetime NEAR execution costs.',M,y,W-2*M,8.7,13.5)
    y-=20
    y=label(c,'Transaction-cost sensitivity',y)
    rows=[(q['scenario'],f"{q['execution_bps_per_hedge']:.0f}",f"{q['rehedges_30d']}",f"{q['spread_vol_points']:.1f}",f"{money(q['dealer_bid_usd'])} / {money(q['dealer_offer_usd'])}") for q in d['tca']]
    y=simple_table(c,['Scenario','Cost (bps)','Hedges','Spread (pts)','ATM bid / offer'],rows,y,[125,88,67,102,119],24,8)
    y-=10
    y=ptext(c,'A dealer selling 10,000 ATM calls would initially buy around 5,677 NEAR to offset short call delta; buying them reverses the initial hedge. A thinner book, funding, basis, larger size, or more rehedging can widen the quote. One book snapshot is not a month of realised hedge costs.',M,y,W-2*M,8.5,13)
    c.showPage()
    # PAGE 3: empirical limits, next steps, conclusion
    header(c,3,at)
    y=title(c,'What we can trust — and what we cannot','Validation, data gaps and how I would quote a client',H-81)
    y=label(c,'Model checks',y)
    y=ptext(c,'<b>Forward-volatility test:</b> the fixed blend scored about 42.8 vol points MAE on the later 25-date holdout; HAR scored 29.6 points, but the outcomes overlap, HAR had negative forecast bias and the live estimate was an outlier. We keep HAR as a challenger. The earlier ~29-point fixed-blend error used a different period and must not be compared directly.',M,y,W-2*M,8.9,14)
    y-=9
    historic=d['historical_iv_future_rv_reference']
    y=ptext(c,f'<b>Historical options test:</b> {historic["n"]} trade-derived peer-IV observations were compared with volatility realised in the subsequent 30 days. Mean IV minus subsequent RV was <b>{historic["mean_iv_minus_future_rv_pts"]:+.1f} vol points</b>. This is too small and overlapping a sample to infer a true risk premium or trading profitability. Historical option trades are not executable two-sided quotes.',M,y,W-2*M,8.9,14)
    y-=19
    extras=d.get('extra_research',{})
    if any(isinstance(v,dict) and v.get('status')=='ok' for v in extras.values()):
        y=label(c,'Supplemental data check (not automatically used for fair IV)',y)
        facts=[]
        h=extras.get('hourly_rv',{})
        if 'close_to_close_hourly_rv_30d_pct' in h:
            facts.append(f"Hourly close-based RV30: {h['close_to_close_hourly_rv_30d_pct']:.1f}% (Coinbase)")
        x=extras.get('extended_forecast_validation',{})
        if x.get('status')=='ok':facts.append(f"Longer-history holdout: {x['n_nonoverlap_holdout']} non-overlapping dates; see CSV for scores")
        perps=extras.get('perp_book_and_funding',{})
        if perps.get('status')=='ok':
            facts.append(f"NEAR perp displayed spread: {perps['quoted_spread_bps']:.1f} bps, ONE book snapshot only")
        if facts:y=ptext(c,'; '.join(facts)+'.',M,y,W-2*M,8.0,12)
        y-=12
    y=label(c,'Key limitations',y)
    limitations=[
      ('Volatility forecast','Daily closes span a limited set of crypto regimes; the fixed blend weights are judgemental. Hourly and longer-history tests need validation.'),
      ('Peer option surface','Only four assets had usable quotes. Marks and thin books may not be executable. The smile fit is not guaranteed globally arbitrage-free.'),
      ('IV versus realised vol','Today\'s IV minus past RV is not the premium earned by selling options. The 11-observation historical test is insufficient.'),
      ('Hedging economics','Transaction costs, funding, perp basis, gamma turnover and jump reserves are illustrative. We lack a realised delta-hedging P&amp;L backtest.'),
      ('Contract and execution','Zero rates/carry, European USD cash settlement and zero starting inventory. Actual RFQ size, expiry, collateral and hedge venue would change the quote.')]
    for head,body in limitations:
        c.setFillColor(NAVY);c.setFont('DS-Bold',8.6);c.drawString(M,y,head)
        y=ptext(c,body,M+138,y+8,W-2*M-138,8.2,12.2)
        y-=11
    y-=10
    y=label(c,'What I would improve with better data',y)
    y=ptext(c,'Collect a longer NEAR price history and completed hourly returns; compare models on separate market regimes with non-overlapping test windows. Build a deeper history of dated altcoin option mids, fit an arbitrage-aware surface, and measure real NEAR perp depth, funding, basis and rehedging slippage before setting a firm spread.',M,y,W-2*M,8.8,13.5)
    y-=20
    y=label(c,'My approach to a client quote',y)
    y=ptext(c,'For a client RFQ, I would first confirm <b>strike, expiry, settlement and size</b>, then use NEAR realised volatility and liquid peer smiles to form a candidate fair IV. I would check it against current market conditions and alternative forecasts rather than accept any single model blindly. Next, I would estimate how easily I can hedge the position, including fees, slippage, funding and jump risk. Finally I would quote <b>both a bid and an offer</b> around fair value, adjust for our existing inventory, and recheck the market before committing. The objective is a defensible two-way price, not a view that volatility must rise or fall.',M,y,W-2*M,9.1,14.5)
    c.showPage();c.save()
    return Path(filepath)
