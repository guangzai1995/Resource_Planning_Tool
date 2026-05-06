import { useState, useEffect, useRef } from 'react'
import {
  Card, Form, Select, Input, InputNumber, Button, Row, Col,
  AutoComplete, Divider,
  Table, Tag, Typography, Badge, Space, message, Tooltip, Popconfirm,
} from 'antd'
import { PlayCircleOutlined, SyncOutlined, InfoCircleOutlined, StopOutlined } from '@ant-design/icons'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  submitBenchmark, listBenchmarkRuns, getBenchmarkResults,
  listGpus, listModels, cancelBenchmark,
  BenchmarkRun, BenchmarkPointResult,
} from '../api'

const { Title, Text } = Typography

const STATUS_MAP: Record<string, { color: string; label: string }> = {
  pending:    { color: 'default',    label: '待运行' },
  running:    { color: 'processing', label: '运行中' },
  cancelling: { color: 'warning',   label: '停止中…' },
  done:       { color: 'success',    label: '已完成' },
  completed:  { color: 'success',    label: '已完成' },
  failed:     { color: 'error',      label: '失败' },
  cancelled:  { color: 'warning',    label: '已取消' },
}

const fmt = (v: number | null | undefined, digits = 1) =>
  v == null ? '–' : v.toFixed(digits)

export default function Benchmark() {
  const [form] = Form.useForm()
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null)
  const [activeStatus, setActiveStatus] = useState<string>('')
  const [logs, setLogs] = useState<string[]>([])
  const logEndRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const qc = useQueryClient()

  const { data: gpus = [] } = useQuery({ queryKey: ['gpus'], queryFn: listGpus })
  const { data: models = [] } = useQuery({ queryKey: ['models'], queryFn: listModels })
  const { data: runs = [], refetch: refetchRuns } = useQuery({
    queryKey: ['benchmark-runs'],
    queryFn: listBenchmarkRuns,
    refetchInterval: activeStatus === 'running' || activeStatus === 'pending' ? 3000 : false,
  })

  // 已完成任务的详细结果
  const { data: results = [] } = useQuery<BenchmarkPointResult[]>({
    queryKey: ['benchmark-results', activeTaskId],
    queryFn: () => getBenchmarkResults(activeTaskId!),
    enabled: !!activeTaskId && activeStatus === 'done',
  })

  const submitMut = useMutation({
    mutationFn: submitBenchmark,
    onSuccess: (data) => {
      message.success('任务已提交')
      setActiveTaskId(data.task_id)
      setActiveStatus('running')
      setLogs([])
      connectWs(data.task_id)
      qc.invalidateQueries({ queryKey: ['benchmark-runs'] })
    },
    onError: () => message.error('提交失败'),
  })

  const cancelMut = useMutation({
    mutationFn: cancelBenchmark,
    onSuccess: () => {
      message.success('已发送停止信号，等待当前测试点完成后退出…')
      setActiveStatus('cancelling')
    },
    onError: () => message.error('停止失败，任务可能已结束'),
  })

  const connectWs = (taskId: string) => {
    if (wsRef.current) wsRef.current.close()
    const ws = new WebSocket(`ws://${window.location.host}/api/v1/benchmark/${taskId}/stream`)
    wsRef.current = ws
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)
        if (msg.type === 'log') {
          setLogs((prev) => [...prev, msg.content])
          logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
        } else if (msg.type === 'end') {
          setActiveStatus(msg.status === 'done' ? 'done' : msg.status)
          qc.invalidateQueries({ queryKey: ['benchmark-runs'] })
          qc.invalidateQueries({ queryKey: ['benchmark-results', taskId] })
        }
      } catch {
        setLogs((prev) => [...prev, e.data])
      }
    }
    ws.onclose = () => {
      qc.invalidateQueries({ queryKey: ['benchmark-runs'] })
    }
  }

  useEffect(() => () => { wsRef.current?.close() }, [])

  const onSubmit = (values: Record<string, unknown>) => {
    const parseTagList = (raw: unknown): number[] => {
      if (Array.isArray(raw)) return (raw as string[]).map((s) => parseInt(String(s))).filter(Boolean)
      return String(raw).split(',').map((s) => parseInt(s.trim())).filter(Boolean)
    }
    submitMut.mutate({
      api_base_url: values.api_base_url as string,
      api_key: values.api_key as string,
      backend_type: values.backend_type as 'openai' | 'openai-chat',
      model_name: values.model_name as string,
      gpu_name: values.gpu_name as string,
      gpu_count: values.gpu_count as number,
      input_tokens_list: parseTagList(values.input_tokens_list),
      output_tokens: values.output_tokens as number,
      concurrency_list: parseTagList(values.concurrency_list),
      epochs: values.epochs as number,
      max_ttft_ms: (values.max_ttft_ms as number) ?? null,
      min_throughput_per_user: (values.min_throughput_per_user as number) ?? null,
      tokenizer_path: (values.tokenizer_path as string) || null,
    })
  }

  // 结果表格列
  const resultColumns = [
    { title: '输入 tokens', dataIndex: 'input_tokens', width: 90 },
    { title: '并发数', dataIndex: 'concurrency', width: 70 },
    {
      title: (
        <Tooltip title="系统级吞吐（所有并发请求合计）">
          吞吐 <small>tok/s</small> <InfoCircleOutlined style={{ fontSize: 11 }} />
        </Tooltip>
      ),
      dataIndex: 'throughput_tokens_s',
      render: (v: number | null) => fmt(v),
      width: 100,
    },
    {
      title: '单用户吞吐 tok/s',
      dataIndex: 'throughput_per_user_tokens_s',
      render: (v: number | null) => fmt(v),
      width: 120,
    },
    { title: 'TTFT均值 ms', dataIndex: 'ttft_mean_ms', render: (v: number | null) => fmt(v), width: 100 },
    { title: 'TTFT P90 ms', dataIndex: 'ttft_p90_ms', render: (v: number | null) => fmt(v), width: 100 },
    { title: 'TTFT P99 ms', dataIndex: 'ttft_p99_ms', render: (v: number | null) => fmt(v), width: 100 },
    { title: '增量延时均值 ms', dataIndex: 'decode_latency_mean_ms', render: (v: number | null) => fmt(v), width: 120 },
    { title: '增量延时 P90 ms', dataIndex: 'decode_latency_p90_ms', render: (v: number | null) => fmt(v), width: 120 },
  ]

  // 历史任务列
  const runColumns = [
    {
      title: '任务 ID',
      dataIndex: 'task_id',
      render: (v: string) => (
        <a onClick={() => {
          const run = runs.find((r) => r.task_id === v)
          setActiveTaskId(v)
          setActiveStatus(run?.status ?? '')
          setLogs([])
          if (run?.status === 'running') connectWs(v)
        }}>
          {v.slice(0, 8)}…
        </a>
      ),
    },
    { title: 'GPU', dataIndex: 'gpu_name' },
    { title: '模型', dataIndex: 'model_name' },
    {
      title: '状态',
      dataIndex: 'status',
      render: (v: string) => {
        const s = STATUS_MAP[v] || { color: 'default', label: v }
        return (
          <Badge
            status={s.color as 'default' | 'processing' | 'success' | 'error' | 'warning'}
            text={s.label}
          />
        )
      },
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      render: (v: string) => new Date(v).toLocaleString('zh-CN'),
    },
    {
      title: '操作',
      render: (_: unknown, r: BenchmarkRun) => (
        <Space>
          <a onClick={() => {
            setActiveTaskId(r.task_id)
            setActiveStatus(r.status)
            setLogs([])
            if (r.status === 'running') connectWs(r.task_id)
            qc.invalidateQueries({ queryKey: ['benchmark-results', r.task_id] })
          }}>
            {r.status === 'done' ? '查看结果' : '查看日志'}
          </a>
        </Space>
      ),
    },
  ]

  const activeRun = runs.find((r) => r.task_id === activeTaskId)

  return (
    <div>
      <Title level={4} style={{ margin: '0 0 16px' }}>基准测试</Title>
      <Row gutter={16}>
        {/* ── 左侧：提交表单 ── */}
        <Col span={8}>
          <Card title="提交任务" size="small">
            <Form
              form={form}
              layout="vertical"
              initialValues={{
                api_key: 'token-abc',
                backend_type: 'openai-chat',
                gpu_count: 8,
                input_tokens_list: ['512', '2048'],
                output_tokens: 512,
                concurrency_list: ['1', '4', '8', '16', '32'],
                epochs: 5,
              }}
              onFinish={onSubmit}
            >
              <Form.Item
                name="api_base_url"
                label="API Base URL"
                rules={[{ required: true, message: '请输入 API 地址' }]}
              >
                <Input placeholder="http://0.0.0.0:9999/v1" />
              </Form.Item>

              <Row gutter={8}>
                <Col span={14}>
                  <Form.Item name="api_key" label="API Key">
                    <Input placeholder="token-abc" />
                  </Form.Item>
                </Col>
                <Col span={10}>
                  <Form.Item name="backend_type" label="接口类型">
                    <Select
                      options={[
                        { value: 'openai-chat', label: 'chat/completions' },
                        { value: 'openai',      label: 'completions' },
                      ]}
                    />
                  </Form.Item>
                </Col>
              </Row>

              <Row gutter={8}>
                <Col span={14}>
                  <Form.Item name="model_name" label="模型" rules={[{ required: true }]}>
                    <AutoComplete
                      placeholder="选择或输入模型名称"
                      options={models.map((m) => ({ value: m.name }))}
                      filterOption={(input, option) =>
                        (option?.value ?? '').toLowerCase().includes(input.toLowerCase())
                      }
                    />
                  </Form.Item>
                </Col>
                <Col span={10}>
                  <Form.Item name="gpu_count" label="GPU 卡数">
                    <Select options={[1, 2, 4, 8].map((v) => ({ value: v, label: `${v} 卡` }))} />
                  </Form.Item>
                </Col>
              </Row>

              <Form.Item name="gpu_name" label="GPU 型号" rules={[{ required: true }]}>
                <Select
                  placeholder="选择 GPU"
                  options={gpus.map((g) => ({ value: g.name, label: g.name }))}
                />
              </Form.Item>

              <Form.Item
                name="input_tokens_list"
                label={
                  <span>
                    输入 tokens 列表&nbsp;
                    <Text type="secondary" style={{ fontSize: 11 }}>(回车或逗号分隔多个值)</Text>
                  </span>
                }
              >
                <Select mode="tags" tokenSeparators={[',']} style={{ width: '100%' }}
                  placeholder="512, 1024, 2048" />
              </Form.Item>

              <Row gutter={8}>
                <Col span={12}>
                  <Form.Item name="output_tokens" label="输出 tokens">
                    <InputNumber min={1} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    name="epochs"
                    label={
                      <Tooltip title="每个测试点的测试轮数。每轮同时发 concurrency 个并发请求，全部完成后进入下一轮。轮数越多统计越稳定，总请求数 = epochs × 并发数">
                        测试轮数（epochs） <InfoCircleOutlined style={{ fontSize: 11 }} />
                      </Tooltip>
                    }
                  >
                    <InputNumber min={1} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
              </Row>

              <Form.Item
                name="concurrency_list"
                label={
                  <span>
                    并发列表&nbsp;
                    <Text type="secondary" style={{ fontSize: 11 }}>(回车或逗号分隔)</Text>
                  </span>
                }
              >
                <Select mode="tags" tokenSeparators={[',']} style={{ width: '100%' }}
                  placeholder="1, 4, 8, 16, 32" />
              </Form.Item>

              <Divider style={{ margin: '8px 0' }} />
              <Row gutter={8}>
                <Col span={12}>
                  <Form.Item
                    name="max_ttft_ms"
                    label={
                      <Tooltip title="首token延时均值超过此阈值时，停止当前输入长度下更高并发的测试。留空表示不限制">
                        最大TTFT <small>ms</small> <InfoCircleOutlined style={{ fontSize: 11 }} />
                      </Tooltip>
                    }
                  >
                    <InputNumber min={0} placeholder="不限制" style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    name="min_throughput_per_user"
                    label={
                      <Tooltip title="单用户吞吐低于此阈值时，停止当前输入长度下更高并发的测试。留空表示不限制">
                        最小单用户吞吐 <small>tok/s</small> <InfoCircleOutlined style={{ fontSize: 11 }} />
                      </Tooltip>
                    }
                  >
                    <InputNumber min={0} placeholder="不限制" style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
              </Row>

              <Form.Item
                name="tokenizer_path"
                label={
                  <Tooltip title="分词器路径或 modelscope/huggingface 模型名（如 Qwen/Qwen2.5-7B-Instruct）。用于精确计算 token 数（包含思考内容）。留空则自动尝试 model_name，失败则回退到 API 返回的 usage 字段">
                    分词器路径 <InfoCircleOutlined style={{ fontSize: 11 }} />
                  </Tooltip>
                }
              >
                <Input placeholder="留空自动尝试（可填 modelscope/HF 模型名）" />
              </Form.Item>

              <Row gutter={8}>
                <Col span={activeStatus === 'running' ? 14 : 24}>
                  <Button
                    type="primary"
                    htmlType="submit"
                    block
                    icon={<PlayCircleOutlined />}
                    loading={submitMut.isPending}
                    disabled={activeStatus === 'running'}
                  >
                    开始测试
                  </Button>
                </Col>
                {activeStatus === 'running' && (
                  <Col span={10}>
                    <Popconfirm
                      title="确认停止测试？"
                      description="当前测试点运行完成后将停止，已收集的数据会保存。"
                      okText="停止"
                      cancelText="继续"
                      okButtonProps={{ danger: true }}
                      onConfirm={() => activeTaskId && cancelMut.mutate(activeTaskId)}
                    >
                      <Button
                        block
                        danger
                        icon={<StopOutlined />}
                        loading={cancelMut.isPending}
                      >
                        停止测试
                      </Button>
                    </Popconfirm>
                  </Col>
                )}
              </Row>
            </Form>
          </Card>
        </Col>

        {/* ── 右侧：日志 + 结果 ── */}
        <Col span={16}>
          {activeTaskId && (
            <Card
              title={`日志 — ${activeTaskId.slice(0, 8)}…`}
              size="small"
              style={{ marginBottom: 16 }}
              extra={
                <Tag
                  color={
                    activeStatus === 'running' ? 'processing' :
                    activeStatus === 'done'    ? 'success' :
                    activeStatus === 'failed'  ? 'error' : 'default'
                  }
                >
                  {STATUS_MAP[activeStatus]?.label ?? activeStatus}
                </Tag>
              }
            >
              <div
                style={{
                  background: '#1a1a1a',
                  color: '#00ff88',
                  fontFamily: 'monospace',
                  fontSize: 12,
                  padding: 12,
                  height: 200,
                  overflowY: 'auto',
                  borderRadius: 4,
                }}
              >
                {logs.length === 0 ? (
                  <Text style={{ color: '#888' }}>等待日志输出…</Text>
                ) : (
                  logs.map((line, i) => (
                    <div key={i} style={{ color: line.startsWith('[ERROR]') ? '#ff4d4f' : line.startsWith('[DONE]') ? '#52c41a' : '#00ff88' }}>
                      {line}
                    </div>
                  ))
                )}
                <div ref={logEndRef} />
              </div>
            </Card>
          )}

          {/* 测试结果表 */}
          {results.length > 0 && (
            <Card
              title={`测试结果 — ${activeRun?.model_name} / ${activeRun?.gpu_name}`}
              size="small"
              style={{ marginBottom: 16 }}
            >
              <Table
                dataSource={results}
                columns={resultColumns}
                rowKey={(r) => `${r.input_tokens}-${r.concurrency}`}
                size="small"
                scroll={{ x: true }}
                pagination={false}
              />
            </Card>
          )}

          {/* 历史任务表 */}
          <Card
            title="历史任务"
            size="small"
            extra={
              <Button size="small" icon={<SyncOutlined />} onClick={() => refetchRuns()}>
                刷新
              </Button>
            }
          >
            <Table
              dataSource={runs}
              columns={runColumns}
              rowKey="task_id"
              size="small"
              pagination={{ pageSize: 8, showSizeChanger: false }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  )
}
