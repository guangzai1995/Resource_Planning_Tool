import { useState } from 'react'
import {
  Card, Button, Upload, Table, Tag, Typography, Row, Col,
  Select, Space, message, Tooltip, Divider,
  Modal, Form, InputNumber, Popconfirm,
} from 'antd'
import { UploadOutlined, ReloadOutlined, EditOutlined, DeleteOutlined } from '@ant-design/icons'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  getDataCoverage, importData, reimportData, listGpus, listModels,
  listBenchmarkData, updateBenchmarkData, deleteBenchmarkData,
  BenchmarkDataRecord,
} from '../api'
import type { UploadFile } from 'antd/es/upload/interface'

const { Title, Text } = Typography

export default function Data() {
  const [gpuFilter, setGpuFilter] = useState<string>()
  const [modelFilter, setModelFilter] = useState<string>()
  const qc = useQueryClient()

  const { data: gpus = [] } = useQuery({ queryKey: ['gpus'], queryFn: listGpus })
  const { data: models = [] } = useQuery({ queryKey: ['models'], queryFn: listModels })
  const { data: coverage, isLoading } = useQuery({
    queryKey: ['coverage', gpuFilter, modelFilter],
    queryFn: () => getDataCoverage({ gpu_name: gpuFilter, model_name: modelFilter }),
  })

  const importMut = useMutation({
    mutationFn: importData,
    onSuccess: (res) => {
      message.success(`导入成功: ${res.rows_imported} 行, ${res.sheets_processed} 个工作表`)
      qc.invalidateQueries({ queryKey: ['coverage'] })
    },
    onError: () => message.error('导入失败'),
  })

  const reimportMut = useMutation({
    mutationFn: reimportData,
    onSuccess: (res) => {
      message.success(`重新导入完成: ${res.rows_imported} 行`)
      qc.invalidateQueries({ queryKey: ['coverage'] })
    },
    onError: () => message.error('重新导入失败'),
  })

  const detailColumns = [
    { title: 'GPU', dataIndex: 'gpu_name' },
    { title: '模型', dataIndex: 'model_name' },
    { title: 'GPU 数量', dataIndex: 'gpu_count' },
    { title: '数据条数', dataIndex: 'data_count' },
    {
      title: '覆盖状态',
      dataIndex: 'data_count',
      render: (v: number) => (
        <Tag color={v === 0 ? 'red' : v < 10 ? 'orange' : 'green'}>
          {v === 0 ? '无数据' : v < 10 ? '数据稀少' : '数据充足'}
        </Tag>
      ),
    },
    {
      title: '输入范围',
      render: (_: unknown, r: { min_input?: number; max_input?: number }) =>
        r.min_input != null ? `${r.min_input} ~ ${r.max_input}` : '-',
    },
    {
      title: '并发范围',
      render: (_: unknown, r: { min_concurrency?: number; max_concurrency?: number }) =>
        r.min_concurrency != null ? `${r.min_concurrency} ~ ${r.max_concurrency}` : '-',
    },
  ]

  return (
    <div>
      <Title level={4} style={{ margin: '0 0 16px' }}>数据管理</Title>

      {/* 导入操作区 */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row gutter={16} align="middle">
          <Col>
            <Upload
              accept=".xlsx,.csv"
              showUploadList={false}
              beforeUpload={(file) => {
                const formData = new FormData()
                formData.append('file', file)
                importMut.mutate(formData)
                return false
              }}
            >
              <Button
                icon={<UploadOutlined />}
                loading={importMut.isPending}
                type="primary"
              >
                上传 Excel/CSV
              </Button>
            </Upload>
          </Col>
          <Col>
            <Tooltip title="从配置的 EXCEL_DATA_PATH 重新导入所有数据">
              <Button
                icon={<ReloadOutlined />}
                loading={reimportMut.isPending}
                onClick={() => reimportMut.mutate()}
              >
                重新导入默认文件
              </Button>
            </Tooltip>
          </Col>
          <Col flex="auto">
            <Text type="secondary" style={{ fontSize: 12 }}>
              支持格式：xlsx（自动识别工作表），csv（单张测试数据）
            </Text>
          </Col>
          <Col>
            <Text>
              总计: <strong>{coverage?.total_rows ?? 0}</strong> 条数据
            </Text>
          </Col>
        </Row>
      </Card>

      {/* 明细表 */}
      <Card size="small" title="覆盖明细" style={{ marginBottom: 16 }}>
        <Table
          dataSource={coverage?.items ?? []}
          columns={detailColumns}
          rowKey={(r) => `${r.gpu_name}-${r.model_name}-${r.gpu_count}`}
          size="small"
          pagination={{ pageSize: 10, showSizeChanger: false }}
          loading={isLoading}
        />
      </Card>

      {/* 数据记录管理 */}
      <RecordsPanel gpus={gpus.map((g) => g.name)} models={models.map((m) => m.name)} />
    </div>
  )
}

// ─── 数据记录管理面板 ───────────────────────────────────────────────────────────

interface RecordsPanelProps {
  gpus: string[]
  models: string[]
}

function RecordsPanel({ gpus, models }: RecordsPanelProps) {
  const qc = useQueryClient()
  const [filterGpu, setFilterGpu] = useState<string>()
  const [filterModel, setFilterModel] = useState<string>()
  const [filterGpuCount, setFilterGpuCount] = useState<number>()
  const [filterInputTokens, setFilterInputTokens] = useState<number>()
  const [page, setPage] = useState(1)
  const pageSize = 50

  const [selectedIds, setSelectedIds] = useState<number[]>([])
  const [editingRecord, setEditingRecord] = useState<BenchmarkDataRecord | null>(null)
  const [editForm] = Form.useForm()

  const { data: records, isLoading } = useQuery({
    queryKey: ['benchmark-records', filterGpu, filterModel, filterGpuCount, filterInputTokens, page],
    queryFn: () => listBenchmarkData({
      gpu_name: filterGpu, model_name: filterModel,
      gpu_count: filterGpuCount, input_tokens: filterInputTokens,
      page, page_size: pageSize,
    }),
  })

  const updateMut = useMutation({
    mutationFn: ({ id, body }: { id: number; body: Partial<BenchmarkDataRecord> }) =>
      updateBenchmarkData(id, body),
    onSuccess: () => {
      message.success('更新成功')
      qc.invalidateQueries({ queryKey: ['benchmark-records'] })
      setEditingRecord(null)
    },
    onError: () => message.error('更新失败'),
  })

  const deleteMut = useMutation({
    mutationFn: (ids: number[]) => deleteBenchmarkData(ids),
    onSuccess: (res) => {
      message.success(`已删除 ${res.deleted} 条记录`)
      setSelectedIds([])
      qc.invalidateQueries({ queryKey: ['benchmark-records'] })
    },
    onError: () => message.error('删除失败'),
  })

  const openEdit = (r: BenchmarkDataRecord) => {
    setEditingRecord(r)
    editForm.setFieldsValue({
      throughput_tokens_s: r.throughput_tokens_s,
      throughput_per_user_tokens_s: r.throughput_per_user_tokens_s,
      ttft_mean_ms: r.ttft_mean_ms,
      ttft_p90_ms: r.ttft_p90_ms,
      ttft_p99_ms: r.ttft_p99_ms,
      ttft_max_ms: r.ttft_max_ms,
      decode_latency_mean_ms: r.decode_latency_mean_ms,
      decode_latency_p90_ms: r.decode_latency_p90_ms,
      decode_latency_p99_ms: r.decode_latency_p99_ms,
      decode_latency_max_ms: r.decode_latency_max_ms,
    })
  }

  const handleEditOk = () => {
    if (!editingRecord) return
    const values = editForm.getFieldsValue()
    updateMut.mutate({ id: editingRecord.id, body: values })
  }

  const columns = [
    { title: 'ID', dataIndex: 'id', width: 60 },
    { title: 'GPU', dataIndex: 'gpu_name', width: 100, ellipsis: true },
    { title: '模型', dataIndex: 'model_name', width: 140, ellipsis: true },
    { title: '卡数', dataIndex: 'gpu_count', width: 60 },
    { title: '输入', dataIndex: 'input_tokens', width: 70 },
    { title: '输出', dataIndex: 'output_tokens', width: 70 },
    { title: '并发', dataIndex: 'concurrency', width: 60 },
    {
      title: '吞吐 tok/s',
      dataIndex: 'throughput_tokens_s',
      width: 90,
      render: (v: number | null) => (v != null ? v.toFixed(1) : '-'),
    },
    {
      title: 'TTFT均值 ms',
      dataIndex: 'ttft_mean_ms',
      width: 100,
      render: (v: number | null) => (v != null ? v.toFixed(1) : '-'),
    },
    {
      title: '增量延时均值 ms',
      dataIndex: 'decode_latency_mean_ms',
      width: 120,
      render: (v: number | null) => (v != null ? v.toFixed(1) : '-'),
    },
    {
      title: '操作',
      width: 110,
      render: (_: unknown, r: BenchmarkDataRecord) => (
        <Space size={4}>
          <Button size="small" icon={<EditOutlined />} onClick={() => openEdit(r)}>编辑</Button>
          <Popconfirm
            title="确认删除该条记录？"
            onConfirm={() => deleteMut.mutate([r.id])}
            okText="删除" cancelText="取消"
          >
            <Button size="small" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ]

  const inputTokenOptions = [128, 256, 512, 1024, 2048, 4096, 8192]

  return (
    <Card size="small" title="数据记录管理">
      {/* 过滤栏 */}
      <Row gutter={8} style={{ marginBottom: 12 }}>
        <Col>
          <Select placeholder="GPU型号" allowClear style={{ width: 150 }}
            onChange={(v) => { setFilterGpu(v); setPage(1) }}
            options={gpus.map((g) => ({ value: g, label: g }))}
          />
        </Col>
        <Col>
          <Select placeholder="模型" allowClear style={{ width: 180 }}
            onChange={(v) => { setFilterModel(v); setPage(1) }}
            options={models.map((m) => ({ value: m, label: m }))}
          />
        </Col>
        <Col>
          <Select placeholder="卡数" allowClear style={{ width: 90 }}
            onChange={(v) => { setFilterGpuCount(v); setPage(1) }}
            options={[1, 2, 4, 8].map((n) => ({ value: n, label: `${n}卡` }))}
          />
        </Col>
        <Col>
          <Select placeholder="输入tokens" allowClear style={{ width: 120 }}
            onChange={(v) => { setFilterInputTokens(v); setPage(1) }}
            options={inputTokenOptions.map((n) => ({ value: n, label: `${n}` }))}
          />
        </Col>
        <Col>
          <Popconfirm
            title={`确认批量删除选中的 ${selectedIds.length} 条记录？`}
            onConfirm={() => deleteMut.mutate(selectedIds)}
            okText="删除" cancelText="取消"
            disabled={selectedIds.length === 0}
          >
            <Button danger disabled={selectedIds.length === 0}>
              批量删除 {selectedIds.length > 0 ? `(${selectedIds.length})` : ''}
            </Button>
          </Popconfirm>
        </Col>
      </Row>

      <Table
        dataSource={records?.items ?? []}
        columns={columns}
        rowKey="id"
        size="small"
        loading={isLoading}
        rowSelection={{
          selectedRowKeys: selectedIds,
          onChange: (keys) => setSelectedIds(keys as number[]),
        }}
        pagination={{
          current: page,
          pageSize,
          total: records?.total ?? 0,
          onChange: (p) => setPage(p),
          showTotal: (t) => `共 ${t} 条`,
          showSizeChanger: false,
        }}
        scroll={{ x: 1100 }}
      />

      {/* 编辑 Modal */}
      <Modal
        title={`编辑记录 #${editingRecord?.id}`}
        open={!!editingRecord}
        onOk={handleEditOk}
        onCancel={() => setEditingRecord(null)}
        confirmLoading={updateMut.isPending}
        width={520}
      >
        <Form form={editForm} layout="vertical" size="small">
          <Row gutter={12}>
            <Col span={12}>
              <Form.Item name="throughput_tokens_s" label="吞吐 tok/s">
                <InputNumber style={{ width: '100%' }} min={0} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="throughput_per_user_tokens_s" label="单用户吞吐 tok/s">
                <InputNumber style={{ width: '100%' }} min={0} />
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={12}>
            <Col span={12}>
              <Form.Item name="ttft_mean_ms" label="TTFT均值 ms">
                <InputNumber style={{ width: '100%' }} min={0} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="ttft_p90_ms" label="TTFT P90 ms">
                <InputNumber style={{ width: '100%' }} min={0} />
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={12}>
            <Col span={12}>
              <Form.Item name="ttft_p99_ms" label="TTFT P99 ms">
                <InputNumber style={{ width: '100%' }} min={0} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="ttft_max_ms" label="TTFT最大值 ms">
                <InputNumber style={{ width: '100%' }} min={0} />
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={12}>
            <Col span={12}>
              <Form.Item name="decode_latency_mean_ms" label="增量延时均值 ms">
                <InputNumber style={{ width: '100%' }} min={0} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="decode_latency_p90_ms" label="增量延时 P90 ms">
                <InputNumber style={{ width: '100%' }} min={0} />
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={12}>
            <Col span={12}>
              <Form.Item name="decode_latency_p99_ms" label="增量延时 P99 ms">
                <InputNumber style={{ width: '100%' }} min={0} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="decode_latency_max_ms" label="增量延时最大值 ms">
                <InputNumber style={{ width: '100%' }} min={0} />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>
    </Card>
  )
}
