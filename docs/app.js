(() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const C = {
    text: "#e8eefc",
    muted: "#93a4c4",
    grid: "rgba(120,160,255,0.12)",
    obs: "#e8eefc",
    m0: "#7c8db5",
    m1: "#4cc9f0",
    m2: "#38d39f",
    m3: "#ff9f1c",
    signed: "#a78bfa",
  };

  const PLOT_CONFIG = { responsive: true, displayModeBar: false };

  function baseLayout(title, xTitle, yTitle, extra = {}) {
    return Object.assign({
      title: { text: title, x: 0.02, xanchor: "left", font: { size: 14, color: C.text } },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: C.muted, family: "Inter, ui-sans-serif, system-ui, sans-serif", size: 11 },
      margin: { l: 58, r: 18, t: 52, b: 66 },
      xaxis: { title: { text: xTitle, font: { size: 11 } }, gridcolor: C.grid, zerolinecolor: C.grid, tickfont: { size: 10 } },
      yaxis: { title: { text: yTitle, font: { size: 11 } }, gridcolor: C.grid, zerolinecolor: C.grid, tickfont: { size: 10 } },
      legend: { orientation: "h", y: -0.24, x: 0, font: { size: 10 }, bgcolor: "rgba(0,0,0,0)" },
      hoverlabel: { bgcolor: "#0c1324", bordercolor: "#2a3a60", font: { color: C.text, size: 11 } },
    }, extra);
  }

  function lineTrace(name, x, y, color, width = 2, dash = "solid") {
    return {
      x, y, name, type: "scatter", mode: "lines+markers",
      line: { color, width, dash, shape: "spline", smoothing: 0.4 },
      marker: { color, size: 5 },
      hovertemplate: `${name}<br>%{x:.2f}, %{y:.3f}<extra></extra>`,
    };
  }

  function renderLoss(syn) {
    const traceTrain = {
      x: syn.models, y: syn.train, name: "train", type: "bar",
      marker: { color: "rgba(76,201,240,0.65)", line: { color: "#4cc9f0", width: 1 } },
      hovertemplate: "train loss = %{y:.3f}<extra></extra>",
    };
    const traceHeld = {
      x: syn.models, y: syn.heldout, name: "held-out", type: "bar",
      marker: { color: "rgba(247,37,133,0.62)", line: { color: "#f72585", width: 1 } },
      hovertemplate: "held-out loss = %{y:.3f}<extra></extra>",
    };
    const layout = baseLayout("候选机制的 train / held-out loss（对数轴）", "", "loss");
    layout.yaxis.type = "log";
    layout.bargap = 0.32;
    layout.annotations = [{
      x: "M1", y: 0.85, text: "best", showarrow: false,
      font: { color: C.m1, size: 11 }, yshift: -14,
    }];
    Plotly.newPlot("lossChart", [traceTrain, traceHeld], layout, PLOT_CONFIG);
  }

  function renderSpeed(syn) {
    const m = syn.mids;
    const sd = syn.speed_density;
    const traces = [
      lineTrace("observed", m, sd.observed, C.obs, 3),
      lineTrace("M0 constant", m, sd.M0_constant, C.m0, 2, "dot"),
      lineTrace("M2 turning", m, sd.M2_turning, C.m2, 2, "dash"),
      lineTrace("M3 instant", m, sd.M3_instant, C.m3, 2, "dash"),
      lineTrace("M1 adaptive", m, sd.M1_adaptive, C.m1, 2.5),
    ];
    const layout = baseLayout("速度-密度关系：观测 vs 四个机制", "local density (cells / area)", "mean speed");
    layout.yaxis.range = [0, 1.1];
    layout.xaxis.range = [0, 0.68];
    Plotly.newPlot("speedChart", traces, layout, PLOT_CONFIG);
  }

  const HEAT_META = {
    single: { title: "孤立单细胞：alpha 不可辨识", note: "loss 在 alpha 方向平坦" },
    population: { title: "群体 · 单一密度：alpha–tau_s 权衡谷", note: "出现斜向退化" },
    two: { title: "群体 · 双密度：谷略变窄", note: "最优点仍接近真值" },
    pulse: { title: "单细胞 + 拥挤脉冲：tau_s 被拉回", note: "下降幅度 + 恢复时间" },
  };

  function renderHeatmap(syn, panel) {
    const idn = syn.identifiability;
    const z = idn.panels[panel];
    const flat = z.flat().filter((v) => Number.isFinite(v));
    const zmax = panel === "single" ? Math.max(...flat) * 1.02 : 6;
    const trace = {
      z, x: idn.alphas, y: idn.taus,
      type: "heatmap", colorscale: "Viridis", zmin: 0, zmax,
      colorbar: {
        title: { text: "loss", side: "right", font: { size: 10 } },
        thickness: 12, len: 0.72, tickfont: { size: 9 },
        outlinewidth: 0,
      },
      hovertemplate: "alpha=%{x:.2f}<br>tau_s=%{y:.2f}<br>loss=%{z:.2f}<extra></extra>",
    };
    const truth = {
      x: [idn.true[0]], y: [idn.true[1]], type: "scatter", mode: "markers",
      marker: { symbol: "x", size: 12, color: "#ffffff", line: { width: 2, color: "#000000" } },
      name: "truth", hoverinfo: "skip", showlegend: false,
    };
    const meta = HEAT_META[panel];
    const layout = baseLayout(meta.title, "alpha (crowding gain)", "tau_s (memory)");
    layout.margin.r = 72;
    layout.margin.b = 70;
    layout.annotations = [{
      xref: "paper", yref: "paper", x: 0.02, y: -0.18,
      text: meta.note, showarrow: false,
      font: { size: 10, color: C.muted },
    }];
    Plotly.newPlot("heatChart", [trace, truth], layout, PLOT_CONFIG);
  }

  function setupHeatTabs(syn) {
    $$("#heatTabs .tab").forEach((btn) => {
      btn.addEventListener("click", () => {
        $$("#heatTabs .tab").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        renderHeatmap(syn, btn.dataset.panel);
      });
    });
  }

  function fmt(x, d = 3) {
    return Number(x).toFixed(d);
  }

  function renderVideo(vid, mids) {
    const m = mids;
    const sim = vid.simulated;
    const traces = [
      lineTrace("observed", m, vid.observed.speed, C.obs, 3),
      lineTrace("M1 adaptive", m, sim.M1_adaptive.speed, C.m1, 2.2),
      lineTrace("M3 instant", m, sim.M3_instant.speed, C.m3, 2.2, "dash"),
      lineTrace("M1 signed", m, sim.M1_signed.speed, C.signed, 2.2, "dot"),
    ];
    const layout = baseLayout("视频速度-密度：观测 vs 拟合", "local density", "mean speed");
    layout.yaxis.range = [0, 1.1];
    layout.xaxis.range = [0, 0.62];
    Plotly.newPlot("videoSpeedChart", traces, layout, PLOT_CONFIG);

    const r = [1, 3, 5, 7, 9, 11];
    const gTraces = [
      lineTrace("observed", r, vid.observed.g, C.obs, 3),
      lineTrace("M1 adaptive", r, sim.M1_adaptive.g, C.m1, 2),
      lineTrace("M3 instant", r, sim.M3_instant.g, C.m3, 2, "dash"),
      lineTrace("M1 signed", r, sim.M1_signed.g, C.signed, 2, "dot"),
    ];
    const gLayout = baseLayout("对关联函数 g(r)：观测 vs 拟合", "r (model units)", "g(r)");
    gLayout.yaxis.range = [0, 2.4];
    gLayout.xaxis.range = [0, 12.5];
    Plotly.newPlot("videoGChart", gTraces, gLayout, PLOT_CONFIG);

    const rows = [
      ["M1_adaptive", "M1（拥挤降速）", vid.fits.M1_adaptive, "s0, tau_s, alpha"],
      ["M1_signed", "M1 signed（允许加速）", vid.fits.M1_signed, "s0, tau_s, alpha"],
      ["M3_instant", "M3（无记忆降速）", vid.fits.M3_instant, "s0, alpha"],
    ];
    rows.sort((a, b) => a[2].loss - b[2].loss);
    const best = rows[0][0];
    $("#fitTable").innerHTML = rows.map(([key, label, fit]) => {
      const p = fit.names.map((n, i) => `${n}=${fmt(fit.vec[i], 3)}`).join(", ");
      const cls = key === best ? "best" : "";
      return `<tr class="${cls}"><td>${label}</td><td>${fmt(fit.loss, 2)}</td><td><code>${p}</code></td></tr>`;
    }).join("");
  }

  function setupExplorer(vid) {
    const frames = vid.frames;
    const img = $("#frameImg");
    const slider = $("#frameSlider");
    const label = $("#frameLabel");
    const playBtn = $("#playBtn");
    let idx = 0;
    let mode = "obs";
    let timer = null;

    slider.max = String(frames.length - 1);

    function show(i) {
      idx = (i + frames.length) % frames.length;
      slider.value = String(idx);
      img.src = frames[idx][mode];
      label.textContent = `t = ${frames[idx].t}`;
    }

    function stop() {
      if (timer) {
        clearInterval(timer);
        timer = null;
        playBtn.textContent = "▶";
      }
    }

    slider.addEventListener("input", (e) => {
      stop();
      show(Number(e.target.value));
    });

    playBtn.addEventListener("click", () => {
      if (timer) {
        stop();
        return;
      }
      playBtn.textContent = "❚❚";
      timer = setInterval(() => show(idx + 1), 420);
    });

    $$("#modePills .pill").forEach((btn) => {
      btn.addEventListener("click", () => {
        mode = btn.dataset.mode;
        $$("#modePills .pill").forEach((b) => b.classList.toggle("active", b === btn));
        show(idx);
      });
    });

    frames.forEach((f) => {
      new Image().src = f.obs;
      new Image().src = f.sim;
    });

    show(0);
  }

  function showError(err) {
    const el = document.createElement("div");
    el.className = "error";
    el.innerHTML = `<strong>页面数据加载失败。</strong><br>${String(err)}<br>
      请确认 <code>docs/assets/data.json</code> 存在，并通过 HTTP 服务器打开（不要直接双击 HTML）。`;
    document.body.prepend(el);
    console.error(err);
  }

  async function main() {
    if (typeof Plotly === "undefined") {
      showError(new Error("Plotly CDN 未加载，请检查网络。"));
      return;
    }
    const res = await fetch("assets/data.json", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    renderLoss(data.synthetic);
    renderSpeed(data.synthetic);
    setupHeatTabs(data.synthetic);
    renderHeatmap(data.synthetic, "single");
    renderVideo(data.video, data.synthetic.mids);
    setupExplorer(data.video);
  }

  document.addEventListener("DOMContentLoaded", () => {
    main().catch(showError);
  });
})();
