import { BrowserRouter, Routes, Route, Navigate, NavLink, useLocation } from 'react-router-dom'
import { Layout, Menu } from 'antd'
import {
  BarChartOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  SettingOutlined,
} from '@ant-design/icons'
import Planner from './pages/Planner'
import Benchmark from './pages/Benchmark'
import Data from './pages/Data'
import Settings from './pages/Settings'

const { Sider, Content } = Layout

// ── Enterprise sidebar ──────────────────────────────────────────────────────
const SIDER_BG = '#0D2255'
const SIDER_SELECTED = '#1664FF'
const SIDER_HOVER = '#1A3680'

const menuItems = [
  {
    key: '/planner',
    icon: <BarChartOutlined />,
    label: <NavLink to="/planner">算力评估</NavLink>,
  },
  {
    key: '/benchmark',
    icon: <ThunderboltOutlined />,
    label: <NavLink to="/benchmark">基准测试</NavLink>,
  },
  {
    key: '/data',
    icon: <DatabaseOutlined />,
    label: <NavLink to="/data">数据管理</NavLink>,
  },
  {
    key: '/settings',
    icon: <SettingOutlined />,
    label: <NavLink to="/settings">设置</NavLink>,
  },
]

function AppContent() {
  const location = useLocation()
  const selectedKey = '/' + location.pathname.split('/')[1]

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider
        width={188}
        style={{
          background: SIDER_BG,
          boxShadow: '2px 0 12px rgba(13,34,85,0.18)',
        }}
      >
        {/* Logo */}
        <div
          style={{
            height: 56,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: '#0A1B3F',
            borderBottom: `1px solid rgba(255,255,255,0.08)`,
          }}
        >
          <span
            style={{
              color: '#FFFFFF',
              fontWeight: 700,
              fontSize: 14,
              letterSpacing: '0.5px',
            }}
          >
            ⬡&nbsp; 算力资源规划
          </span>
        </div>

        <Menu
          mode="inline"
          selectedKeys={[selectedKey]}
          items={menuItems}
          style={{
            background: 'transparent',
            borderRight: 0,
            marginTop: 8,
            color: 'rgba(255,255,255,0.75)',
          }}
          theme="dark"
          // Override dark theme token inline via style
        />

        {/* Bottom version */}
        <div
          style={{
            position: 'absolute',
            bottom: 16,
            left: 0,
            right: 0,
            textAlign: 'center',
            fontSize: 11,
            color: 'rgba(255,255,255,0.25)',
          }}
        >
          v2.1.0
        </div>
      </Sider>

      <Layout>
        {/* Top header bar */}
        <div
          style={{
            height: 48,
            background: '#FFFFFF',
            borderBottom: '1px solid #D6E4FF',
            display: 'flex',
            alignItems: 'center',
            padding: '0 24px',
            boxShadow: '0 1px 4px rgba(22,100,255,0.08)',
          }}
        >
          <span style={{ color: '#4A6080', fontSize: 12 }}>
            {menuItems.find((m) => m.key === selectedKey)?.key === '/planner'
              ? '资源规划工具 / 算力评估'
              : menuItems.find((m) => m.key === selectedKey)?.key === '/benchmark'
              ? '资源规划工具 / 基准测试'
              : menuItems.find((m) => m.key === selectedKey)?.key === '/data'
              ? '资源规划工具 / 数据管理'
              : '资源规划工具 / 设置'}
          </span>
        </div>

        <Content
          style={{
            padding: 20,
            background: '#EEF3FC',
            overflowY: 'auto',
            minHeight: 0,
          }}
        >
          <Routes>
            <Route path="/" element={<Navigate to="/planner" replace />} />
            <Route path="/planner" element={<Planner />} />
            <Route path="/benchmark" element={<Benchmark />} />
            <Route path="/data" element={<Data />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </Content>
      </Layout>
    </Layout>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  )
}

// suppress unused var warning for SIDER_HOVER
void SIDER_SELECTED
void SIDER_HOVER

