import { useState } from 'react'
import {
  Card, Table, Button, Modal, Form, Input, InputNumber,
  Typography, Space, Popconfirm, message, Divider, Row, Col,
} from 'antd'
import { PlusOutlined, EditOutlined, DeleteOutlined } from '@ant-design/icons'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import {
  listGpus, createGpu, updateGpu, deleteGpu,
  listModels, createModel, updateModel, deleteModel,
  GpuSpec, Model,
} from '../api'

const { Title } = Typography

function GpuTable() {
  const [modalOpen, setModalOpen] = useState(false)
  const [editing, setEditing] = useState<GpuSpec | null>(null)
  const [form] = Form.useForm()
  const qc = useQueryClient()

  const { data: gpus = [], isLoading } = useQuery({ queryKey: ['gpus'], queryFn: listGpus })

  const createMut = useMutation({ mutationFn: createGpu, onSuccess: () => { qc.invalidateQueries({ queryKey: ['gpus'] }); setModalOpen(false) } })
  const updateMut = useMutation({ mutationFn: ({ id, data }: { id: number; data: Partial<GpuSpec> }) => updateGpu(id, data), onSuccess: () => { qc.invalidateQueries({ queryKey: ['gpus'] }); setModalOpen(false) } })
  const deleteMut = useMutation({ mutationFn: deleteGpu, onSuccess: () => qc.invalidateQueries({ queryKey: ['gpus'] }) })

  const openCreate = () => { setEditing(null); form.resetFields(); setModalOpen(true) }
  const openEdit = (r: GpuSpec) => { setEditing(r); form.setFieldsValue(r); setModalOpen(true) }

  const onOk = async () => {
    const values = await form.validateFields()
    if (editing) {
      updateMut.mutate({ id: editing.id!, data: values })
    } else {
      createMut.mutate(values)
    }
  }

  const columns = [
    { title: '名称', dataIndex: 'name' },
    { title: '显存 (GB)', dataIndex: 'memory_gb' },
    { title: '带宽 (GB/s)', dataIndex: 'memory_bandwidth_gbps' },
    { title: 'TFLOPS (BF16)', dataIndex: 'tflops_bf16' },
    { title: '价格 (¥/h)', dataIndex: 'price_per_hour' },
    {
      title: '操作',
      render: (_: unknown, r: GpuSpec) => (
        <Space>
          <Button size="small" icon={<EditOutlined />} onClick={() => openEdit(r)}>编辑</Button>
          <Popconfirm title="确定删除?" onConfirm={() => deleteMut.mutate(r.id!)}>
            <Button size="small" danger icon={<DeleteOutlined />}>删除</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <>
      <Card
        size="small"
        title="GPU 规格"
        extra={<Button size="small" icon={<PlusOutlined />} type="primary" onClick={openCreate}>新增</Button>}
      >
        <Table dataSource={gpus} columns={columns} rowKey="id" size="small" loading={isLoading} pagination={false} />
      </Card>

      <Modal
        title={editing ? '编辑 GPU' : '新增 GPU'}
        open={modalOpen}
        onOk={onOk}
        onCancel={() => setModalOpen(false)}
        confirmLoading={createMut.isPending || updateMut.isPending}
      >
        <Form form={form} layout="vertical">
          <Form.Item name="name" label="名称" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
          <Row gutter={8}>
            <Col span={12}>
              <Form.Item name="memory_gb" label="显存 (GB)" rules={[{ required: true }]}>
                <InputNumber min={0} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="memory_bandwidth_gbps" label="带宽 (GB/s)" rules={[{ required: true }]}>
                <InputNumber min={0} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={8}>
            <Col span={12}>
              <Form.Item name="tflops_bf16" label="TFLOPS BF16" rules={[{ required: true }]}>
                <InputNumber min={0} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="price_per_hour" label="价格 (¥/h)" rules={[{ required: true }]}>
                <InputNumber min={0} step={0.1} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>
    </>
  )
}

function ModelTable() {
  const [modalOpen, setModalOpen] = useState(false)
  const [editing, setEditing] = useState<Model | null>(null)
  const [form] = Form.useForm()
  const qc = useQueryClient()

  const { data: models = [], isLoading } = useQuery({ queryKey: ['models'], queryFn: listModels })
  const createMut = useMutation({ mutationFn: createModel, onSuccess: () => { qc.invalidateQueries({ queryKey: ['models'] }); setModalOpen(false) } })
  const updateMut = useMutation({ mutationFn: ({ id, data }: { id: number; data: Partial<Model> }) => updateModel(id, data), onSuccess: () => { qc.invalidateQueries({ queryKey: ['models'] }); setModalOpen(false) } })
  const deleteMut = useMutation({ mutationFn: deleteModel, onSuccess: () => qc.invalidateQueries({ queryKey: ['models'] }) })

  const openCreate = () => { setEditing(null); form.resetFields(); setModalOpen(true) }
  const openEdit = (r: Model) => { setEditing(r); form.setFieldsValue(r); setModalOpen(true) }
  const onOk = async () => {
    const values = await form.validateFields()
    if (editing) { updateMut.mutate({ id: editing.id!, data: values }) }
    else { createMut.mutate(values) }
  }

  const columns = [
    { title: '名称', dataIndex: 'name' },
    { title: '参数量 (B)', dataIndex: 'parameter_b' },
    { title: '类型', dataIndex: 'model_type' },
    { title: '默认路径', dataIndex: 'default_model_path', ellipsis: true },
    {
      title: '操作',
      render: (_: unknown, r: Model) => (
        <Space>
          <Button size="small" icon={<EditOutlined />} onClick={() => openEdit(r)}>编辑</Button>
          <Popconfirm title="确定删除?" onConfirm={() => deleteMut.mutate(r.id!)}>
            <Button size="small" danger icon={<DeleteOutlined />}>删除</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ]

  return (
    <>
      <Card
        size="small"
        title="模型配置"
        extra={<Button size="small" icon={<PlusOutlined />} type="primary" onClick={openCreate}>新增</Button>}
        style={{ marginTop: 16 }}
      >
        <Table dataSource={models} columns={columns} rowKey="id" size="small" loading={isLoading} pagination={false} />
      </Card>

      <Modal
        title={editing ? '编辑模型' : '新增模型'}
        open={modalOpen}
        onOk={onOk}
        onCancel={() => setModalOpen(false)}
        confirmLoading={createMut.isPending || updateMut.isPending}
      >
        <Form form={form} layout="vertical">
          <Form.Item name="name" label="模型名称" rules={[{ required: true }]}>
            <Input placeholder="例: Qwen2.5-32B-Instruct" />
          </Form.Item>
          <Row gutter={8}>
            <Col span={12}>
              <Form.Item name="parameter_b" label="参数量 (B)" rules={[{ required: true }]}>
                <InputNumber min={0} step={0.1} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="model_type" label="类型">
                <Input placeholder="dense / moe" />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item name="default_model_path" label="默认模型路径">
            <Input placeholder="/model/Qwen2.5-32B-Instruct" />
          </Form.Item>
        </Form>
      </Modal>
    </>
  )
}

export default function Settings() {
  return (
    <div>
      <Title level={4} style={{ margin: '0 0 16px' }}>元数据设置</Title>
      <GpuTable />
      <ModelTable />
    </div>
  )
}
