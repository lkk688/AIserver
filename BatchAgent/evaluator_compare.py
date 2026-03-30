import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _generate_speed_tables(speed_data: List[Dict[str, Any]]) -> str:
    html_out = ""
    case_names = sorted(list(set(r["case_name"] for r in speed_data)))
    
    for case in case_names:
        case_data = [r for r in speed_data if r["case_name"] == case]
        html_out += f"<h3>Comparison Tables: {case}</h3>"
        
        # 1. Single Request Results
        single_res_rows = []
        for prefill in sorted(list(set(r["prefill_target"] for r in case_data))):
            matches = [r for r in case_data if r["prefill_target"] == prefill and r["batch_size"] == 1]
            if matches:
                # Use the 'same' prompt baseline if available for single request
                same_matches = [m for m in matches if m["prompt_type"] == "same"]
                m = same_matches[0] if same_matches else matches[0]
                
                single_res_rows.append(
                    f"<tr>"
                    f"<td><strong>pp{prefill}/tg128</strong></td>"
                    f"<td>{m['mean_ttft_ms']}</td>"
                    f"<td>{m['mean_tpot_ms']}</td>"
                    f"<td>{m['pp_tps']} tok/s</td>"
                    f"<td><strong>{m['tg_tps']} tok/s</strong></td>"
                    f"<td>{m['e2e_latency_sec']}s</td>"
                    f"<td>{m['throughput']} tok/s</td>"
                    f"<td>{m.get('peak_mem', 'N/A')}</td>"
                    f"</tr>"
                )
        
        if single_res_rows:
            html_out += f"""
            <div class="benchmark-card">
              <div class="benchmark-card-header">
                <div>⚡ SINGLE REQUEST RESULTS</div>
              </div>
              <table class="benchmark-table">
                <thead>
                  <tr>
                    <th>Test</th>
                    <th>TTFT (ms)</th>
                    <th>TPOT (ms/tok)</th>
                    <th>pp TPS</th>
                    <th>tg TPS</th>
                    <th>E2E Latency</th>
                    <th>Throughput</th>
                    <th>Peak Mem</th>
                  </tr>
                </thead>
                <tbody>
                  {"".join(single_res_rows)}
                </tbody>
              </table>
            </div>
            """
        
        # 2. Continuous batching for each prefill target
        for prefill in sorted(list(set(r["prefill_target"] for r in case_data))):
            for prompt_type in ["same", "different"]:
                matches = [r for r in case_data if r["prefill_target"] == prefill and r["prompt_type"] == prompt_type]
                if not matches:
                    continue
                matches.sort(key=lambda x: x["batch_size"])
                
                baseline = next((r for r in matches if r["batch_size"] == 1), None)
                baseline_tg = baseline["tg_tps"] if baseline and baseline["tg_tps"] > 0 else 1.0
                
                rows_html = []
                for m in matches:
                    bz = m["batch_size"]
                    speedup = m["tg_tps"] / baseline_tg
                    
                    label = f"{bz}x"
                    if bz == 1:
                        label = f"{bz}x (baseline)"
                    
                    pp_req = m["pp_tps"] / bz if bz > 0 else 0
                    
                    rows_html.append(
                        f"<tr>"
                        f"<td>{label}</td>"
                        f"<td><strong>{m['tg_tps']} tok/s</strong></td>"
                        f"<td>{speedup:.2f}x</td>"
                        f"<td>{m['pp_tps']} tok/s</td>"
                        f"<td>{pp_req:.1f} tok/s</td>"
                        f"<td>{m['mean_ttft_ms']}</td>"
                        f"<td>{m['e2e_latency_sec']}s</td>"
                        f"</tr>"
                    )
                
                header_title = f"CONTINUOUS BATCHING — {prompt_type.upper()} PROMPT{'S' if prompt_type == 'different' else ''}"
                header_meta = f"pp{prefill} / tg128"
                if prompt_type == "same":
                    header_meta += " · partial prefix cache hit"
                else:
                    header_meta += " · no cache reuse"
                
                html_out += f"""
                <div class="benchmark-card">
                  <div class="benchmark-card-header">
                    <div>{header_title}</div>
                    <div class="header-meta">{header_meta}</div>
                  </div>
                  <table class="benchmark-table">
                    <thead>
                      <tr>
                        <th>Batch Size</th>
                        <th>tg TPS</th>
                        <th>Speedup</th>
                        <th>pp TPS</th>
                        <th>pp TPS/req</th>
                        <th>Avg TTFT (ms)</th>
                        <th>E2E Latency</th>
                      </tr>
                    </thead>
                    <tbody>
                      {"".join(rows_html)}
                    </tbody>
                  </table>
                </div>
                """
                
    return html_out


def _generate_speed_charts(speed_data: List[Dict[str, Any]]) -> str:
    """
    speed_data is a list of run records.
    """
    if not speed_data:
        return ""

    prefills = sorted(list(set(r["prefill_target"] for r in speed_data)))
    html_sections = []
    
    for prefill in prefills:
        subset = [r for r in speed_data if r["prefill_target"] == prefill]
        group_keys = set(f"{r['case_name']} ({r['prompt_type']})" for r in subset)
        
        throughput_traces = []
        ttft_traces = []
        
        for gkey in sorted(list(group_keys)):
            records = [r for r in subset if f"{r['case_name']} ({r['prompt_type']})" == gkey]
            records.sort(key=lambda x: x["batch_size"])
            
            x_vals = [r["batch_size"] for r in records]
            y_tp = [r["throughput"] for r in records]
            y_ttft = [r["mean_ttft_ms"] for r in records]
            
            throughput_traces.append({
                "x": x_vals,
                "y": y_tp,
                "mode": "lines+markers",
                "name": gkey,
                "type": "scatter"
            })
            
            ttft_traces.append({
                "x": x_vals,
                "y": y_ttft,
                "mode": "lines+markers",
                "name": gkey,
                "type": "scatter"
            })
            
        chart_id_tp = f"chr_tp_{prefill}"
        chart_id_ttft = f"chr_ttft_{prefill}"
        
        divs = f"""
        <div class="chart-card"><div id="{chart_id_tp}"></div></div>
        <div class="chart-card"><div id="{chart_id_ttft}"></div></div>
        <script>
            Plotly.newPlot('{chart_id_tp}', {json.dumps(throughput_traces)}, 
                {{title: 'Throughput (Prefill: {prefill})', xaxis: {{title: 'Batch Size'}}, yaxis: {{title: 'Tokens/sec'}}, paper_bgcolor: 'white', plot_bgcolor: 'white', margin: {{t: 50, b: 50, l: 50, r: 20}} }});
            Plotly.newPlot('{chart_id_ttft}', {json.dumps(ttft_traces)}, 
                {{title: 'TTFT (Prefill: {prefill})', xaxis: {{title: 'Batch Size'}}, yaxis: {{title: 'TTFT (ms)'}}, paper_bgcolor: 'white', plot_bgcolor: 'white', margin: {{t: 50, b: 50, l: 50, r: 20}} }});
        </script>
        """
        html_sections.append(divs)
        
    return "<h2>Speed Benchmark Figures</h2><div class='grid'>" + "".join(html_sections) + "</div>"


def _generate_accuracy_charts(acc_data: List[Dict[str, Any]]) -> str:
    if not acc_data:
        return ""
        
    benchmarks = sorted(list(set(r["benchmark"] for r in acc_data)))
    case_names = sorted(list(set(r["case_name"] for r in acc_data)))
    
    traces = []
    for case in case_names:
        y_vals = []
        for b in benchmarks:
            matches = [r for r in acc_data if r["case_name"] == case and r["benchmark"] == b]
            score = matches[0]["score"] if matches and matches[0].get("score") not in ["", None] else None
            y_vals.append(score)
            
        traces.append({
            "x": benchmarks,
            "y": y_vals,
            "type": "bar",
            "name": case
        })
        
    div = f"""
    <div class="chart-card full"><div id="acc_compare_chart"></div></div>
    <script>
        Plotly.newPlot('acc_compare_chart', {json.dumps(traces)}, 
            {{title: 'Accuracy Comparison', barmode: 'group', xaxis: {{title: 'Benchmark'}}, yaxis: {{title: 'Score (%)'}}, paper_bgcolor: 'white', plot_bgcolor: 'white', margin: {{t: 50, b: 50, l: 50, r: 20}} }});
    </script>
    """
    
    return "<h2>Accuracy Benchmark Comparison</h2><div class='grid'>" + div + "</div>"


def generate_comparison_report(json_files: List[str], output_path: str):
    speed_records = []
    acc_records = []
    
    for fpath_str in json_files:
        fpath = Path(fpath_str).resolve()
        if not fpath.exists():
            logger.warning(f"File not found: {fpath}")
            continue
            
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {fpath}: {e}")
            continue
            
        suite = data.get("suite", "")
        if suite == "speed":
            speed_records.extend(data.get("records", []))
        elif suite == "accuracy":
            acc_records.extend(data.get("records", []))
        else:
            logger.warning(f"Unknown suite in {fpath}: {suite}")
            
    if not speed_records and not acc_records:
        logger.error("No valid datarecords loaded for comparison.")
        return
        
    speed_tables_html = _generate_speed_tables(speed_records)
    speed_html = _generate_speed_charts(speed_records)
    acc_html = _generate_accuracy_charts(acc_records)
    
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LLM Verification Comparison</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      background: #fafafa;
      color: #333;
    }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 32px 24px; }}
    h1, h2, h3 {{ margin: 16px 0 24px 0; font-weight: 700; color: #222; }}
    h1 {{ font-size: 28px; border-bottom: 2px solid #eaeaea; padding-bottom: 12px; }}
    h2 {{ font-size: 22px; margin-top: 40px; }}
    h3 {{ font-size: 18px; color: #555; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-bottom: 32px; }}
    .chart-card {{ background: white; border: 1px solid #eaeaea; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); }}
    .full {{ grid-column: 1 / -1; }}
    
    /* Benchmark Tables */
    .benchmark-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
      background: white;
    }}
    .benchmark-table th, .benchmark-table td {{
      padding: 14px 16px;
      border-bottom: 1px solid #f0f0f0;
      text-align: right;
      color: #333;
    }}
    .benchmark-table th:first-child, .benchmark-table td:first-child {{
      text-align: left;
    }}
    .benchmark-table th {{
      color: #666;
      font-weight: 600;
      text-transform: capitalize;
      font-size: 13px;
    }}
    .benchmark-card {{
      background: white;
      border: 1px solid #eaeaea;
      border-radius: 8px;
      margin-bottom: 24px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.02);
      overflow: hidden;
    }}
    .benchmark-card-header {{
      background: #fdfdfd;
      padding: 12px 16px;
      border-bottom: 1px solid #eaeaea;
      font-weight: 700;
      color: #555;
      text-transform: uppercase;
      font-size: 13px;
      letter-spacing: 0.5px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .benchmark-card-header .header-meta {{
      font-weight: 400;
      color: #999;
      text-transform: none;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>LLM API Comparison Report</h1>
    {speed_tables_html}
    {speed_html}
    {acc_html}
  </div>
</body>
</html>
    """
    
    out_path = Path(output_path).resolve()
    out_path.write_text(html, encoding="utf-8")
    logger.info(f"Comparison report written to {out_path}")

