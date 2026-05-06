import React from 'react'
import ReactDOM from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ConfigProvider, theme } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import App from './App'
import 'antd/dist/reset.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 30000,
    },
  },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <ConfigProvider
        locale={zhCN}
        theme={{
          algorithm: theme.defaultAlgorithm,
          token: {
            colorPrimary: '#1664FF',
            colorLink: '#1664FF',
            colorBgLayout: '#EEF3FC',
            colorBgContainer: '#FFFFFF',
            colorBorder: '#C6D5F5',
            colorBorderSecondary: '#E0EAFF',
            borderRadius: 6,
            borderRadiusLG: 8,
            colorText: '#0A1B3F',
            colorTextSecondary: '#4A6080',
            colorTextTertiary: '#7A90B3',
            fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
          },
          components: {
            Card: {
              headerBg: '#F4F8FF',
              headerFontSize: 13,
            },
            Slider: {
              trackBg: '#1664FF',
              handleColor: '#1664FF',
              dotActiveBorderColor: '#1664FF',
            },
            Select: {
              optionSelectedBg: '#EBF3FF',
            },
            Tag: {
              defaultBg: '#EBF3FF',
              defaultColor: '#1664FF',
            },
          },
        }}
      >
        <App />
      </ConfigProvider>
    </QueryClientProvider>
  </React.StrictMode>
)

