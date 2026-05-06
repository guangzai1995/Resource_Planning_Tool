/**
 * 算力评估页面 — 参照「资源规划工具.xlsx · 算力评估3.0」
 *
 * 计算逻辑完全对齐 Excel 公式：
 *   K (单实例满足并发数) = 最后一个满足双约束点与下一点之间的插值
 *     约束1: per_user_tpt(含网络) >= 需求吞吐
 *     约束2: ttft_pure_s            <= 要求延时 - 网络延时
 *   I (输出单用户吞吐量)    = 在 K 处插值 per_user_tpt(含网络)
 *   J (输出首Token延时)      = 在 K 处插值 ttft_pure_s  + 网络延时
 *   L (所需GPU卡数量)        = CEILING(目标并发 / K) × 卡数/实例
 */
import { useState, useEffect, useRef } from 'react'
import {
  Card, Select, Slider, Row, Col, Typography, Spin, Space, Divider, Tag, Tooltip,
} from 'antd'
import {
  CheckCircleFilled, CloseCircleFilled, MinusCircleFilled, InfoCircleOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import ReactECharts from 'echarts-for-react'
import { listGpus, listModels, predictSweep, getDataOptions } from '../api'
import type { SweepPoint, SweepRequest, DataOptionsItem } from '../api'

const { Text } = Typography

// ── Design tokens ───────────────────────────────────────────────────────────
const C = {
  blue:       '#1664FF',
  blueMid:    '#4880FF',
  blueLight:  '#EBF3FF',
  blueBorder: '#91CAFF',
  navyText:   '#0A1B3F',
  grayText:   '#4A6080',
  okBg:       '#F6FFED',
  okBorder:   '#B7EB8F',
  okText:     '#389E0D',
  failBg:     '#FFF1F0',
  failBorder: '#FFA39E',
  failText:   '#CF1322',
  neutralBg:  '#F4F8FF',
  neutralBorder: '#D6E4FF',
  neutralText: '#8C8C8C',
  chartGreen: '#52C41A',
  chartAmber: '#FAAD14',
  chartRed:   '#FF4D4F',
}

// ── Fixed discrete option table (输出长度) ─────────────────────────────────
const OUTPUT_TOK_OPTS = [128, 256, 512, 1024, 2048]
const OUTPUT_TOK_MARKS: Record<number, string> = { 0:'128', 1:'256', 2:'512', 3:'1K', 4:'2K' }

/** 将 input_tokens 数值转为 slider mark 标签 */
function tokLabel(v: number): string {
  return v >= 1000 ? `${Math.round(v / 1024)}K` : String(v)
}

// ── Source badge ────────────────────────────────────────────────────────────
const SRC_COLOR: Record<string, string> = {
  interpolation: 'blue', ensemble: 'geekblue', model_based: 'orange', unavailable: 'red',
}
const SRC_LABEL: Record<string, string> = {
  interpolation: '实测插值', ensemble: '混合预测', model_based: '理论估算', unavailable: '无数据',
}

// ── Helpers ──────────────────────────────────────────────────────────────────
function fmt(v: number | null | undefined, dec = 1): string {
  if (v == null) return '—'
  return v.toFixed(dec)
}

/**
 * 对齐 Excel K 公式：在 sweep 数据中找到最大安全并发（含插值）
 * 约束：per_user >= minTpt  AND  ttft_pure_s <= maxTtftPure
 *
 * per_user 计算对齐 Excel：= 1000 / (decode_latency_mean_ms + net_lat_ms)
 * 其中 net_lat_ms = netLatS * 1000
 * 若 decode_latency_mean_ms 不可用则回退到 throughput_per_user / (1 + utp*netLatS)
 */
function calcK(
  sweep: SweepPoint[],
  minTpt: number,        // 单用户吞吐量需求 tokens/s（对应 Excel F 列）
  maxTtftReq: number,    // 首Token延时要求 s（含网络，对应 Excel G 列）
  netLatS: number,       // 网络延时 s（对应 Excel H 列）
): number {
  if (sweep.length === 0) return 0

  const maxTtftPure = maxTtftReq - netLatS   // 去掉网络延时后的纯服务延时要求
  const netLatMs = netLatS * 1000

  // Excel 公式：per_user = 1000 / (decode_latency_ms + net_lat_ms)
  // decode_latency_ms 优先使用实测值，不可用时回退到 throughput/concurrency 换算
  const perUser = (p: SweepPoint): number => {
    const dl = p.decode_latency_mean_ms
    if (dl != null && dl > 0) return 1000 / (dl + netLatMs)
    const utp = p.throughput_per_user_tokens_s ?? 0
    return utp / (1 + utp * netLatS)
  }

  // 仅使用实测范围内的真实数据点（非外推）参与 K 计算
  // 外推点因并发数被 clamp 到边界值，会人为制造水平线，导致 K 虚高
  const hasFlagInfo = sweep.some(p => p.is_extrapolation !== undefined)
  const realSweep = hasFlagInfo ? sweep.filter(p => !p.is_extrapolation) : sweep
  if (realSweep.length === 0) return 0

  // Find last index where both constraints satisfied
  let gIdx = -1
  for (let i = 0; i < realSweep.length; i++) {
    const pu   = perUser(realSweep[i])
    const ttft = realSweep[i].ttft_s ?? Infinity
    if (pu >= minTpt && ttft <= maxTtftPure) gIdx = i
  }
  if (gIdx < 0) return 0

  const g = realSweep[gIdx]
  if (gIdx >= realSweep.length - 1) return g.concurrency  // already at max real data boundary

  const h = realSweep[gIdx + 1]
  const per_g = perUser(g)
  const per_h = perUser(h)
  const ft_g  = g.ttft_s ?? 0
  const ft_h  = h.ttft_s ?? 0
  const cg = g.concurrency
  const ch = h.concurrency

  // thrCand: concurrency where throughput = minTpt (interpolated)
  let thrCand: number
  if (per_h >= minTpt)          thrCand = ch
  else if (per_h === per_g)     thrCand = cg
  else thrCand = cg + (minTpt - per_g) * (ch - cg) / (per_h - per_g)

  // latCand: concurrency where ttft = maxTtftPure (interpolated)
  let latCand: number
  if (ft_h <= maxTtftPure)      latCand = ch
  else if (ft_h === ft_g)       latCand = cg
  else latCand = cg + (maxTtftPure - ft_g) * (ch - cg) / (ft_h - ft_g)

  if (ch === cg) return cg
  // K = FLOOR(MAX(g, MIN(h, MIN(thrCand, latCand))), 1)
  return Math.floor(Math.max(cg, Math.min(ch, Math.min(thrCand, latCand))))
}

/**
 * 在 sweep 曲线上对 concurrency=target 做线性插值
 * 返回插值后的 decode_latency_mean_ms（对应 Excel 数据表 L 列），用于计算 I
 */
function interpAt(sweep: SweepPoint[], target: number): { decLatMs: number | null; ttftS: number | null; source: string | null } {
  if (sweep.length === 0) return { decLatMs: null, ttftS: null, source: null }

  let lo = sweep[0]
  let hi = sweep[sweep.length - 1]
  for (let i = 0; i < sweep.length - 1; i++) {
    if (sweep[i].concurrency <= target && sweep[i + 1].concurrency >= target) {
      lo = sweep[i]; hi = sweep[i + 1]; break
    }
  }

  const t = lo.concurrency === hi.concurrency ? 0
    : (target - lo.concurrency) / (hi.concurrency - lo.concurrency)

  // 插值 decode_latency_mean_ms（与 Excel 数据表 L 列对应）
  const dl_lo = lo.decode_latency_mean_ms ?? null
  const dl_hi = hi.decode_latency_mean_ms ?? null
  const decLatMs = dl_lo != null && dl_hi != null ? dl_lo + t * (dl_hi - dl_lo) : null

  const ttft_lo = lo.ttft_s ?? null
  const ttft_hi = hi.ttft_s ?? null
  const ttftS = ttft_lo != null && ttft_hi != null ? ttft_lo + t * (ttft_hi - ttft_lo) : null

  return { decLatMs, ttftS, source: lo.source }
}

// ────────────────────────────────────────────────────────────────────────────
export default function Planner() {
  // ── State ─────────────────────────────────────────────────────────────────
  const [gpuName,        setGpuName]        = useState('')
  const [modelName,      setModelName]      = useState('')
  const [gpuCount,       setGpuCount]       = useState(1)    // 卡数（直接值）
  const [inputTok,       setInputTok]       = useState(512)  // 输入 tokens（直接值）
  const [outputTokIdx,   setOutputTokIdx]   = useState(3)    // 1024 (index into OUTPUT_TOK_OPTS)
  const [targetConc,     setTargetConc]     = useState(32)
  const [minThroughput,  setMinThroughput]  = useState(10)   // tokens/s / user
  const [maxTtftS,       setMaxTtftS]       = useState(5.0)  // s (首token延时要求，含网络)
  const [netLatencyMs,   setNetLatencyMs]   = useState(5)    // ms
  const [sweepReq,       setSweepReq]       = useState<SweepRequest | null>(null)
  const debRef = useRef<ReturnType<typeof setTimeout>>()

  // ── Remote data ───────────────────────────────────────────────────────────
  const { data: gpus   = [] } = useQuery({ queryKey: ['gpus'],   queryFn: listGpus })
  const { data: models = [] } = useQuery({ queryKey: ['models'], queryFn: listModels })
  const { data: dataOptions = [] } = useQuery<DataOptionsItem[]>({
    queryKey: ['dataOptions'],
    queryFn: getDataOptions,
    staleTime: 300000,
  })

  // ── 级联推导：仅显示有实测数据的选项 ────────────────────────────────────
  const gpuOpts = [...new Set(dataOptions.map((c) => c.gpu_name))].sort()
  const modelOpts = [...new Set(
    dataOptions.filter((c) => !gpuName || c.gpu_name === gpuName).map((c) => c.model_name)
  )].sort()
  const gpuCountOpts = [...new Set(
    dataOptions
      .filter((c) => (!gpuName || c.gpu_name === gpuName) && (!modelName || c.model_name === modelName))
      .map((c) => c.gpu_count)
  )].sort((a, b) => a - b)
  const selectedCombo = dataOptions.find(
    (c) => c.gpu_name === gpuName && c.model_name === modelName && c.gpu_count === gpuCount
  )
  const inputTokOpts: number[] = selectedCombo?.input_tokens ?? []

  // GPU index in dynamic array (for slider)
  const gpuCountSliderIdx = Math.max(0, gpuCountOpts.indexOf(gpuCount))
  const inputTokSliderIdx = Math.max(0, inputTokOpts.indexOf(inputTok))
  const gpuCountMarks: Record<number, string> = Object.fromEntries(
    gpuCountOpts.map((v, i) => [i, String(v)])
  )
  const inputTokMarks: Record<number, string> = Object.fromEntries(
    inputTokOpts.map((v, i) => [i, tokLabel(v)])
  )

  // ── 级联 useEffect：上级变化时自动修正下级到第一个有效值 ─────────────────
  // 初始化：dataOptions 加载后设置 GPU
  useEffect(() => {
    if (gpuOpts.length > 0 && !gpuName) setGpuName(gpuOpts[0])
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gpuOpts.length])

  // GPU 变化 → 重置 modelName
  useEffect(() => {
    if (modelOpts.length > 0 && !modelOpts.includes(modelName)) setModelName(modelOpts[0])
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gpuName, modelOpts.join(',')])

  // Model 变化 → 重置 gpuCount
  useEffect(() => {
    if (gpuCountOpts.length > 0 && !gpuCountOpts.includes(gpuCount)) setGpuCount(gpuCountOpts[0])
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelName, gpuCountOpts.join(',')])

  // gpuCount 变化 → 重置 inputTok（取中间值）
  useEffect(() => {
    if (inputTokOpts.length > 0 && !inputTokOpts.includes(inputTok)) {
      setInputTok(inputTokOpts[Math.floor(inputTokOpts.length / 2)])
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gpuCount, inputTokOpts.join(',')])

  // Debounced sweep trigger
  // 注意：仅在影响 API 请求的参数变化时触发（netLatencyMs / minThroughput / maxTtftS
  // 是纯前端计算参数，不进入请求体，它们变化时图表通过响应式计算立即更新）
  useEffect(() => {
    if (!gpuName || !modelName) return
    clearTimeout(debRef.current)
    debRef.current = setTimeout(() => {
      setSweepReq({
        gpu_name:      gpuName,
        model_name:    modelName,
        gpu_count:     gpuCount,
        input_tokens:  inputTok,
        output_tokens: OUTPUT_TOK_OPTS[outputTokIdx],
        max_concurrency: 512,  // 固定范围：覆盖大多数实测数据，不随目标并发变化
      })
    }, 600)
    return () => clearTimeout(debRef.current)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gpuName, modelName, gpuCount, inputTok, outputTokIdx])

  const { data: sweep = [], isFetching } = useQuery({
    queryKey: ['sweep', sweepReq],
    queryFn:  () => predictSweep(sweepReq!),
    enabled:  sweepReq !== null,
    staleTime: 60000,
  })

  // ── Core calculations (Excel formula) ────────────────────────────────────
  const netLatS = netLatencyMs / 1000
  const gpuCountPerInst = gpuCount

  // K — 单实例满足并发数（Excel 公式 K 列）
  const K = calcK(sweep, minThroughput, maxTtftS, netLatS)

  // L — 所需GPU卡数量（Excel 公式 L 列）: CEILING(目标并发/K) × 卡数/实例
  const L: number | null = K > 0 ? Math.ceil(targetConc / K) * gpuCountPerInst : null

  // I — 在 K 处的单用户吞吐量（含网络）（Excel 公式 I 列）
  // J — 在 K 处的首Token延时 + 网络延时（Excel 公式 J 列）
  const atK = K > 0 ? interpAt(sweep, K) : { decLatMs: null, ttftS: null, source: null }
  // Excel 公式 I 列：1000 / (decode_latency_ms + net_lat_ms)
  const I: number | null = atK.decLatMs != null && atK.decLatMs > 0
    ? 1000 / (atK.decLatMs + netLatencyMs)
    : null
  const J: number | null = atK.ttftS != null ? atK.ttftS + netLatS : null

  // Status checks
  const iOk = I != null && I >= minThroughput
  const jOk = J != null && J <= maxTtftS
  const kOk = K > 0
  const lOk = L != null

  const dominantSource = sweep.length ? (sweep[Math.floor(sweep.length / 2)]?.source ?? null) : null

  // ── Chart data ────────────────────────────────────────────────────────────
  // 只取实测范围内的数据点绘图（过滤外推点），x 轴范围自然止于实测边界
  const hasFlagInfo = sweep.some(p => p.is_extrapolation !== undefined)
  const chartPoints = hasFlagInfo ? sweep.filter(p => !p.is_extrapolation) : sweep

  // 外推区警告：目标并发超过实测数据范围
  const maxRealConc = chartPoints.length > 0 ? chartPoints[chartPoints.length - 1].concurrency : 0
  const targetBeyondData = maxRealConc > 0 && targetConc > maxRealConc

  // per_user per chart point（对齐 Excel：1000/(decode_latency_ms+net_lat_ms)）
  const perAdjData = chartPoints.map((p) => {
    const dl = p.decode_latency_mean_ms
    const adj = dl != null && dl > 0
      ? 1000 / (dl + netLatencyMs)
      : (() => { const utp = p.throughput_per_user_tokens_s ?? 0; return utp / (1 + utp * netLatS) })()
    const color = (p.ttft_s ?? Infinity) + netLatS > maxTtftS
      ? C.chartRed
      : adj < minThroughput ? C.chartAmber : C.chartGreen
    return { value: [p.concurrency, +adj.toFixed(2)], itemStyle: { color } }
  })

  const ttftData = chartPoints.map((p) => {
    const total = p.ttft_s != null ? +(p.ttft_s + netLatS).toFixed(3) : null
    const ok = total != null && total <= maxTtftS
    return { value: [p.concurrency, total], itemStyle: { color: ok ? C.chartGreen : C.chartRed } }
  })

  const chartGrid = { left: 62, right: 20, top: 38, bottom: 46 }
  // xAxis 使用 value 类型，markLine 可精确定位（无需字符串匹配）
  const chartXAxis = {
    type: 'value' as const,
    name: '并发数',
    nameLocation: 'end' as const,
    nameTextStyle: { fontSize: 11, color: C.grayText },
    axisLabel: { fontSize: 11 },
    axisLine: { lineStyle: { color: C.neutralBorder } },
    axisTick: { lineStyle: { color: C.neutralBorder } },
    minInterval: 1,
  }

  // markLine 直接用数值（value 轴无需字符串匹配）
  const markLines = (reqLine: { axis: 'y' | 'x'; val: number; label: string; color: string }[]) =>
    reqLine.map((m) => ({
      [m.axis === 'y' ? 'yAxis' : 'xAxis']: m.val,
      lineStyle: { color: m.color, type: 'dashed', width: 1.5 },
      label: { formatter: m.label, color: m.color, fontSize: 10 },
    }))

  const throughputChartOpt = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis' as const,
      formatter: (params: any[]) => {
        const item = params[0]
        const x = Array.isArray(item.value) ? item.value[0] : item.axisValue
        const y = Array.isArray(item.value) ? item.value[1] : item.value
        return `并发数: ${x}<br/>单用户吞吐量: <b>${fmt(y, 1)}</b> tokens/s`
      },
    },
    grid: chartGrid,
    xAxis: chartXAxis,
    yAxis: { type: 'value', name: 'tokens/s', nameTextStyle: { fontSize: 11, color: C.grayText }, axisLabel: { fontSize: 11 } },
    series: [{
      name: '单用户吞吐量',
      type: 'line',
      smooth: true,
      data: perAdjData,
      symbolSize: 5,
      lineStyle: { width: 2, color: C.blue },
      markLine: {
        silent: true, symbol: 'none',
        data: markLines([
          { axis: 'y', val: minThroughput, label: `需≥${minThroughput}`, color: C.chartRed },
          ...(K > 0 ? [{ axis: 'x' as const, val: K, label: `K=${K}`, color: C.blue }] : []),
        ]),
      },
    }],
  }

  const ttftChartOpt = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis' as const,
      formatter: (params: any[]) => {
        const item = params[0]
        const x = Array.isArray(item.value) ? item.value[0] : item.axisValue
        const y = Array.isArray(item.value) ? item.value[1] : item.value
        return `并发数: ${x}<br/>首Token延时（含网络）: <b>${fmt(y, 3)}</b> s`
      },
    },
    grid: chartGrid,
    xAxis: chartXAxis,
    yAxis: { type: 'value', name: '延时 (s)', nameTextStyle: { fontSize: 11, color: C.grayText }, axisLabel: { fontSize: 11 } },
    series: [{
      name: '首Token延时',
      type: 'line',
      smooth: true,
      data: ttftData,
      symbolSize: 5,
      lineStyle: { width: 2, color: C.blueMid },
      markLine: {
        silent: true, symbol: 'none',
        data: markLines([
          { axis: 'y', val: maxTtftS,     label: `需≤${maxTtftS}s`, color: C.chartRed },
          ...(K > 0 ? [{ axis: 'x' as const, val: K, label: `K=${K}`, color: C.blue }] : []),
        ]),
      },
    }],
  }

  // ── KPI 卡片配置（对应 Excel 输出列）────────────────────────────────────
  const kpis = [
    {
      key: 'I', label: '单用户吞吐量', sub: `需 ≥ ${minThroughput} tokens/s`,
      value: I, unit: 'tokens/s', dec: 1, ok: iOk,
      tip: 'Excel I 列：在K处插值，含网络延时影响',
    },
    {
      key: 'J', label: '平均首 Token 延时', sub: `需 ≤ ${maxTtftS} s（含网络）`,
      value: J, unit: 's', dec: 3, ok: jOk,
      tip: 'Excel J 列：在K处插值平均TTFT + 网络延时',
    },
    {
      key: 'K', label: '单实例满足并发数', sub: `目标 ${targetConc} 并发`,
      value: K > 0 ? K : null, unit: '个', dec: 0, ok: kOk,
      tip: 'Excel K 列：双约束插值法求最大安全并发',
    },
    {
      key: 'L', label: '所需 GPU 卡数量', sub: `CEILING(${targetConc} / K) × ${gpuCountPerInst}`,
      value: L, unit: '卡', dec: 0, ok: lOk,
      tip: 'Excel L 列：= CEILING(目标并发 / K) × 卡数/实例',
    },
  ]

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <Row gutter={16} style={{ minHeight: '100%', alignItems: 'flex-start' }}>

      {/* ── Left: 输入参数 ─────────────────────────────────────────────── */}
      <Col xs={24} md={8} lg={7}>
        <Card
          size="small"
          title={
            <span style={{ color: C.navyText, fontWeight: 600 }}>
              输入参数
            </span>
          }
          style={{ border: `1px solid ${C.neutralBorder}` }}
          styles={{ header: { borderBottom: `1px solid ${C.neutralBorder}` }, body: { padding: '14px 16px' } }}
        >
          <Space direction="vertical" style={{ width: '100%' }} size={4}>

            {/* GPU 卡型号 */}
            <div>
              <Text style={{ fontSize: 12, color: C.grayText }}>GPU 卡型号</Text>
              <Select
                value={gpuName || undefined}
                onChange={setGpuName}
                style={{ width: '100%', marginTop: 4 }}
                options={gpuOpts.map((name) => {
                  const spec = gpus.find((g) => g.name === name)
                  return { value: name, label: spec ? `${name}（${spec.memory_gb} GB）` : name }
                })}
                placeholder="选择 GPU"
              />
            </div>

            {/* 模型参数量 */}
            <div style={{ marginTop: 6 }}>
              <Text style={{ fontSize: 12, color: C.grayText }}>模型参数量 (B) / 型号</Text>
              <Select
                value={modelName || undefined}
                onChange={setModelName}
                style={{ width: '100%', marginTop: 4 }}
                options={modelOpts.map((name) => {
                  const spec = models.find((m) => m.name === name)
                  return { value: name, label: spec ? `${name}（${spec.parameter_b}B）` : name }
                })}
                placeholder="选择模型"
              />
            </div>

            <Divider style={{ margin: '10px 0 4px', borderColor: C.neutralBorder }} />

            {/* 单个服务占用GPU卡数量 */}
            <SliderRow
              label="单个服务占用 GPU 卡数量"
              value={`${gpuCount} 卡`}
            >
              <Slider min={0} max={Math.max(0, gpuCountOpts.length - 1)} step={1}
                value={gpuCountSliderIdx}
                onChange={(idx) => setGpuCount(gpuCountOpts[idx] ?? gpuCount)}
                marks={gpuCountMarks}
                tooltip={{ formatter: (v?: number) => v != null ? `${gpuCountOpts[v] ?? ''}卡` : '' }}
                style={{ marginBottom: 16 }}
                disabled={gpuCountOpts.length === 0}
              />
            </SliderRow>

            {/* 模型输入上下文长度 */}
            <SliderRow
              label="模型输入上下文长度 (tokens)"
              value={inputTok > 0 ? inputTok.toLocaleString() : '—'}
            >
              <Slider min={0} max={Math.max(0, inputTokOpts.length - 1)} step={1}
                value={inputTokSliderIdx}
                onChange={(idx) => setInputTok(inputTokOpts[idx] ?? inputTok)}
                marks={inputTokMarks}
                tooltip={{ formatter: (v?: number) => v != null ? `${(inputTokOpts[v] ?? 0).toLocaleString()} tokens` : '' }}
                style={{ marginBottom: 16 }}
                disabled={inputTokOpts.length === 0}
              />
            </SliderRow>

            {/* 输出长度 */}
            <SliderRow label="输出长度 (tokens)" value={String(OUTPUT_TOK_OPTS[outputTokIdx])}>
              <Slider min={0} max={OUTPUT_TOK_OPTS.length - 1} step={1}
                value={outputTokIdx} onChange={setOutputTokIdx}
                marks={OUTPUT_TOK_MARKS}
                tooltip={{ formatter: (v?: number) => v != null ? `${OUTPUT_TOK_OPTS[v]} tokens` : '' }}
                style={{ marginBottom: 16 }}
              />
            </SliderRow>

            {/* 目标并发数 */}
            <SliderRow label="目标并发数" value={String(targetConc)}>
              <Slider min={1} max={1024} step={1}
                value={targetConc} onChange={setTargetConc}
                marks={{ 1: '1', 256: '256', 512: '512', 1024: '1024' }}
                tooltip={{ formatter: (v?: number) => v != null ? `${v}` : '' }}
                style={{ marginBottom: 16 }}
              />
            </SliderRow>

            <Divider style={{ margin: '4px 0', borderColor: C.neutralBorder }} />
            <Text style={{ fontSize: 11, color: C.grayText }}>单实例服务推理速度需求</Text>

            {/* 单用户吞吐量需求 */}
            <SliderRow label="单用户吞吐量 ≥" value={`${minThroughput} tokens/s`}>
              <Slider min={5} max={20} step={1}
                value={minThroughput} onChange={setMinThroughput}
                marks={{ 5: '5', 10: '10', 15: '15', 20: '20' }}
                tooltip={{ formatter: (v?: number) => v != null ? `${v} t/s` : '' }}
                style={{ marginBottom: 16 }}
              />
            </SliderRow>

            {/* 首token延时要求 */}
            <SliderRow label="首 Token 延时 ≤" value={`${maxTtftS.toFixed(1)} s`}>
              <Slider min={1} max={10} step={0.5}
                value={maxTtftS} onChange={setMaxTtftS}
                marks={{ 1: '1s', 5: '5s', 10: '10s' }}
                tooltip={{ formatter: (v?: number) => v != null ? `${v} s` : '' }}
                style={{ marginBottom: 16 }}
              />
            </SliderRow>

            <Divider style={{ margin: '4px 0', borderColor: C.neutralBorder }} />
            <Text style={{ fontSize: 11, color: C.grayText }}>网络延时</Text>

            {/* 网络延时 */}
            <SliderRow label="网络延时 (s)" value={`${netLatS.toFixed(3)} s`}>
              <Slider min={0} max={100} step={1}
                value={netLatencyMs} onChange={setNetLatencyMs}
                marks={{ 0: '0', 5: '5ms', 20: '20ms', 100: '100ms' }}
                tooltip={{ formatter: (v?: number) => v != null ? `${v} ms` : '' }}
                style={{ marginBottom: 4 }}
              />
            </SliderRow>

          </Space>
        </Card>
      </Col>

      {/* ── Right: 结果输出 + 可视化 ──────────────────────────────────── */}
      <Col xs={24} md={16} lg={17}>
        <Spin spinning={isFetching} tip="正在预测…">

          {/* ── 结果输出区（Excel 输出列 I J K L）─────────────────────── */}
          <div style={{ marginBottom: 14 }}>
            <div style={{ marginBottom: 8, display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 13, fontWeight: 600, color: C.navyText }}>结果输出</span>
              {dominantSource && (
                <Tag color={SRC_COLOR[dominantSource] ?? 'default'} style={{ fontSize: 11 }}>
                  {SRC_LABEL[dominantSource] ?? dominantSource}
                </Tag>
              )}
              {targetBeyondData && (
                <Tag color="warning" style={{ fontSize: 11 }}>
                  目标并发 {targetConc} 超出实测数据范围（最大 {maxRealConc}），K 取实测边界值
                </Tag>
              )}
            </div>
            <Row gutter={12}>
              {kpis.map((c) => {
                const hasVal = c.value != null
                const bg     = !hasVal ? C.neutralBg  : c.ok ? C.okBg    : C.failBg
                const border = !hasVal ? C.neutralBorder : c.ok ? C.okBorder : C.failBorder
                const valCol = !hasVal ? C.neutralText : c.ok ? C.okText   : C.failText
                const Icon   = !hasVal ? MinusCircleFilled : c.ok ? CheckCircleFilled : CloseCircleFilled
                const icoCol = !hasVal ? '#D9D9D9'    : c.ok ? C.okText   : C.failText

                return (
                  <Col span={6} key={c.key}>
                    <Card
                      size="small"
                      style={{ background: bg, borderColor: border, textAlign: 'center' }}
                      styles={{ body: { padding: '10px 6px 8px' } }}
                    >
                      <div style={{ fontSize: 24, fontWeight: 700, lineHeight: 1.2, color: valCol }}>
                        {fmt(c.value, c.dec)}
                        <span style={{ fontSize: 11, fontWeight: 400, marginLeft: 2 }}>{c.unit}</span>
                      </div>
                      <div style={{ fontSize: 12, color: C.navyText, marginTop: 4, fontWeight: 500 }}>
                        {c.label}{' '}
                        <Tooltip title={c.tip}>
                          <InfoCircleOutlined style={{ fontSize: 10, color: C.grayText }} />
                        </Tooltip>
                      </div>
                      <div style={{ fontSize: 10, color: C.grayText, marginTop: 2 }}>{c.sub}</div>
                      <Icon style={{ color: icoCol, marginTop: 5, fontSize: 13 }} />
                    </Card>
                  </Col>
                )
              })}
            </Row>
          </div>

          {/* ── 可视化图表区 ─────────────────────────────────────────── */}
          <Row gutter={12}>
            <Col span={12}>
              <Card
                size="small"
                title={
                  <span style={{ fontSize: 12, color: C.navyText }}>
                    单用户吞吐量 vs 并发数
                    <span style={{ fontSize: 10, color: C.grayText, marginLeft: 6, fontWeight: 400 }}>
                      🟢达标 · 🟡吞吐不足 · 🔴延时超限
                    </span>
                  </span>
                }
                style={{ border: `1px solid ${C.neutralBorder}` }}
                styles={{ header: { borderBottom: `1px solid ${C.neutralBorder}` } }}
              >
                <ReactECharts option={throughputChartOpt} style={{ height: 250 }} notMerge />
              </Card>
            </Col>
            <Col span={12}>
              <Card
                size="small"
                title={
                  <span style={{ fontSize: 12, color: C.navyText }}>
                    首 Token 延时 vs 并发数
                    <span style={{ fontSize: 10, color: C.grayText, marginLeft: 6, fontWeight: 400 }}>
                      含网络延时 {netLatencyMs} ms
                    </span>
                  </span>
                }
                style={{ border: `1px solid ${C.neutralBorder}` }}
                styles={{ header: { borderBottom: `1px solid ${C.neutralBorder}` } }}
              >
                <ReactECharts option={ttftChartOpt} style={{ height: 250 }} notMerge />
              </Card>
            </Col>
          </Row>

          {sweep.length === 0 && !isFetching && (
            <div style={{ textAlign: 'center', padding: '56px 0', color: C.grayText }}>
              请在左侧选择模型和 GPU 型号，图表将自动更新
            </div>
          )}

          {/* ── 并发与性能关系图表（总吞吐 + 增量延时）─────────────── */}
          {sweep.length > 0 && (
            <Row gutter={12} style={{ marginTop: 12 }}>
              <Col span={12}>
                <Card
                  size="small"
                  title={
                    <span style={{ fontSize: 12, color: C.navyText }}>
                      总吞吐量 vs 并发数
                      <span style={{ fontSize: 10, color: C.grayText, marginLeft: 6, fontWeight: 400 }}>
                        tokens/s（整个实例）
                      </span>
                    </span>
                  }
                  style={{ border: `1px solid ${C.neutralBorder}` }}
                  styles={{ header: { borderBottom: `1px solid ${C.neutralBorder}` } }}
                >
                  <ReactECharts
                    option={{
                      backgroundColor: 'transparent',
                      tooltip: {
                        trigger: 'axis' as const,
                        formatter: (params: any[]) => {
                          const item = params[0]
                          const x = Array.isArray(item.value) ? item.value[0] : item.axisValue
                          const y = Array.isArray(item.value) ? item.value[1] : item.value
                          return `并发数: ${x}<br/>总吞吐量: <b>${y != null ? (+y).toFixed(1) : '—'}</b> tokens/s`
                        },
                      },
                      grid: chartGrid,
                      xAxis: chartXAxis,
                      yAxis: { type: 'value', name: 'tokens/s', nameTextStyle: { fontSize: 11, color: C.grayText }, axisLabel: { fontSize: 11 } },
                      series: [{
                        name: '总吞吐量',
                        type: 'line',
                        smooth: true,
                        symbolSize: 5,
                        lineStyle: { width: 2, color: '#13C2C2' },
                        areaStyle: { color: 'rgba(19,194,194,0.08)' },
                        data: chartPoints.map((p) => [p.concurrency, p.throughput_tokens_s]),
                        markLine: {
                          silent: true, symbol: 'none',
                          data: K > 0 ? [{ xAxis: K, lineStyle: { color: C.blue, type: 'dashed', width: 1.5 }, label: { formatter: `K=${K}`, color: C.blue, fontSize: 10 } }] : [],
                        },
                      }],
                    }}
                    style={{ height: 250 }}
                    notMerge
                  />
                </Card>
              </Col>
              <Col span={12}>
                <Card
                  size="small"
                  title={
                    <span style={{ fontSize: 12, color: C.navyText }}>
                      平均增量延时 vs 并发数
                      <span style={{ fontSize: 10, color: C.grayText, marginLeft: 6, fontWeight: 400 }}>
                        每 token 解码均值
                      </span>
                    </span>
                  }
                  style={{ border: `1px solid ${C.neutralBorder}` }}
                  styles={{ header: { borderBottom: `1px solid ${C.neutralBorder}` } }}
                >
                  <ReactECharts
                    option={{
                      backgroundColor: 'transparent',
                      tooltip: {
                        trigger: 'axis' as const,
                        formatter: (params: any[]) => {
                          const item = params[0]
                          const x = Array.isArray(item.value) ? item.value[0] : item.axisValue
                          const y = Array.isArray(item.value) ? item.value[1] : item.value
                          return `并发数: ${x}<br/>平均增量延时: <b>${y != null ? (+y).toFixed(2) : '—'}</b> ms`
                        },
                      },
                      grid: chartGrid,
                      xAxis: chartXAxis,
                      yAxis: { type: 'value', name: '延时 (ms)', nameTextStyle: { fontSize: 11, color: C.grayText }, axisLabel: { fontSize: 11 } },
                      series: [{
                        name: '平均增量延时',
                        type: 'line',
                        smooth: true,
                        symbolSize: 5,
                        lineStyle: { width: 2, color: '#722ED1' },
                        areaStyle: { color: 'rgba(114,46,209,0.07)' },
                        data: chartPoints.map((p) => [p.concurrency, p.decode_latency_mean_ms]),
                        markLine: {
                          silent: true, symbol: 'none',
                          data: K > 0 ? [{ xAxis: K, lineStyle: { color: C.blue, type: 'dashed', width: 1.5 }, label: { formatter: `K=${K}`, color: C.blue, fontSize: 10 } }] : [],
                        },
                      }],
                    }}
                    style={{ height: 250 }}
                    notMerge
                  />
                </Card>
              </Col>
            </Row>
          )}
        </Spin>
      </Col>
    </Row>
  )
}

// ── 小工具组件：带右侧数值标签的 Slider Row ──────────────────────────────
function SliderRow({
  label,
  value,
  children,
}: {
  label: string
  value: string
  children: React.ReactNode
}) {
  return (
    <div>
      <Row justify="space-between" align="middle" style={{ marginBottom: 2 }}>
        <Text style={{ fontSize: 12, color: C.grayText }}>{label}</Text>
        <Text strong style={{ fontSize: 13, color: C.blue }}>{value}</Text>
      </Row>
      {children}
    </div>
  )
}
