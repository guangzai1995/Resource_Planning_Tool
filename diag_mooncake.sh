#!/usr/bin/env bash
# ============================================================================
# diag_mooncake.sh —— mooncake / RDMA / PD-disaggregation 现场诊断脚本
#
# 背景: pd-sglang-minimax-0705-test6 (2p2d) 中, prefill(10.200.1.13) 反复抛
#   KVTransferError: Failed to send kv chunk to 10.200.1.14:14129
#   KVTransferError: Decode instance could be dead, remote mooncake session
# 导致 decode 拿不到 KV、生成 0 token, 进而 bench 吞吐=0 触发 auto_bench
# 大面积跳过。本脚本采集判定该故障所需的一切现场证据。
#
# 用法:
#   sudo bash diag_mooncake.sh            # 推荐 root 跑(否则 ib/docker 受限)
#   bash diag_mooncake.sh                 # 非 root 也可, 会降级并标注
#
# 输出:
#   当前目录下生成 diag_<host>_<时间戳>/ 目录, 内含各分类 .txt + SUMMARY.txt
#
# 兼容: CentOS 7 / bash4 / 仅 python2; 全程只读, 不修改任何系统状态。
# ============================================================================
set -uo pipefail

# ----------------- 目标拓扑(来自本次配置) -----------------
PREFILL_IP="10.200.1.13"
DECODE_IP="10.200.1.14"
MC_MASTER_PORT="50051"          # MOONCAKE_MASTER
MC_META_PORT="8080"             # MOONCAKE_TE_META_DATA_SERVER
IB_DEV="mlx5_0"                 # MOONCAKE_DEVICE / disaggregation-ib-device
CONTAINER_PAT="bench-pd"        # 用于定位本次 PD 相关容器(含已停的)
MC_PORT_HINTS="14129 19377 13661 25791"  # 日志里出现过失败的 mooncake 数据端口示例

# ----------------- 输出目录 -----------------
TS="$(date +%Y%m%d_%H%M%S)"
HN="$(hostname 2>/dev/null || echo host)"
OUTDIR="$PWD/diag_${HN}_${TS}"
mkdir -p "$OUTDIR" || { echo "无法创建输出目录 $OUTDIR" >&2; exit 1; }
SUMMARY="$OUTDIR/SUMMARY.txt"
: > "$SUMMARY"

# ----------------- 小工具 -----------------
have() { command -v "$1" >/dev/null 2>&1; }
export -f have   # 导出给 section() 里的 bash -c 子 shell(否则 have 在子 shell 不可见)

# 把一段标题写入 SUMMARY
note() { printf '%s\n' "$*" >> "$SUMMARY"; }

# 自动判定: PASS / FAIL / WARN
#   check "<项>" <0|1|2> "<详情>"
check() {
    local label="$1" st="$2" detail="$3" tag
    case "$st" in
        0) tag="[ PASS ]";;
        1) tag="[ FAIL ]";;
        *) tag="[ WARN ]";;
    esac
    printf '%-44s %s  %s\n' "$label" "$tag" "$detail" >> "$SUMMARY"
}

# 采集一个分类到单独文件; 用法: section "<文件名>" bash -c '<脚本>'
section() {
    local name="$1"; shift
    local out="$OUTDIR/${name}.txt"
    {
        echo "############################################################"
        echo "# $name"
        echo "# date : $(date '+%F %T')"
        echo "# host : $(hostname 2>/dev/null)   ips: $(hostname -I 2>/dev/null)"
        echo "############################################################"
        "$@"
        echo
        echo "[section exit=$?]"
    } > "$out" 2>&1
}

# sudo 包装(若当前非 root 且有免密 sudo 则用)
SUDO=""
[ "$(id -u)" -eq 0 ] || { have sudo && sudo -n true 2>/dev/null && SUDO="sudo"; }

# ----------------- 角色自检 -----------------
MY_IPS="$(hostname -I 2>/dev/null)"
ROLE="unknown"; PEER_IP=""; PEER_ROLE=""
if echo "$MY_IPS" | grep -qw "$DECODE_IP";  then ROLE="decode(14)";  PEER_IP="$PREFILL_IP"; PEER_ROLE="prefill(13)";
elif echo "$MY_IPS" | grep -qw "$PREFILL_IP"; then ROLE="prefill(13)"; PEER_IP="$DECODE_IP";  PEER_ROLE="decode(14)";
fi
{
    echo "# mooncake / RDMA / PD 诊断汇总"
    echo "# 生成时间: $(date '+%F %T')"
    echo "# 主机    : $(hostname)   ips: $MY_IPS"
    echo "# 角色    : $ROLE   (对端: $PEER_ROLE $PEER_IP)"
    echo "# 运行身份: $(id 2>/dev/null | cut -d, -f1)   sudo=${SUDO:-<none>}"
    echo "# IB 设备 : $IB_DEV   master=${DECODE_IP}:${MC_MASTER_PORT}  meta=http://${DECODE_IP}:${MC_META_PORT}/metadata"
    echo
    echo "------ 关键判定 ------"
} >> "$SUMMARY"

echo ">> 输出目录: $OUTDIR"
echo ">> 本机角色: $ROLE  对端: $PEER_ROLE $PEER_IP"
echo ">> 身份: $(id -un)  sudo=${SUDO:-<none>}"
echo

# ============================================================================
# 01 系统 / 内核 / 发行版
# ============================================================================
section "01_system" bash -c '
    echo "#### uname ####"; uname -a
    echo; echo "#### release ####"
    for f in /etc/redhat-release /etc/os-release; do [ -r "$f" ] && { echo "-- $f"; cat "$f"; }; done
    echo; echo "#### uptime / who ####"; uptime; whoami; id
    echo; echo "#### selinux ####"
    if have getenforce; then getenforce; else echo "getenforce NOT FOUND"; fi
    [ -r /etc/selinux/config ] && grep -vE "^\s*#|^\s*$" /etc/selinux/config
    echo; echo "#### nproc / load ####"; nproc; cat /proc/loadavg
'

# ============================================================================
# 02 网络 / 监听端口 / 防火墙
#    重点: 50051(mooncake master) / 8080(metadata) / mooncake 动态高端口
# ============================================================================
section "02_network" bash -c '
    echo "#### ip addr ####"; ip -o addr show 2>/dev/null || ifconfig 2>/dev/null
    echo; echo "#### 路由 (到对端) ####"; ip route 2>/dev/null
    echo; echo "#### TCP 监听 (全部) ####"
    if have ss; then ss -tlnp 2>/dev/null || ss -tln 2>/dev/null
    else netstat -tlnp 2>/dev/null || netstat -tln 2>/dev/null; fi
    echo; echo "==== 只看 mooncake/master/meta/动态端口 ===="
    if have ss; then ss -tlnp 2>/dev/null | grep -E "50051|8080|mooncake|14129|19377|13661|25791" || echo "(无匹配)"; fi
    echo; echo "#### /proc/sys net 关键项 ####"
    for k in net.ipv4.ip_local_port_range net.ipv4.ip_forward net.core.rmem_max net.core.wmem_max; do
        sysctl "$k" 2>/dev/null
    done
    echo; echo "#### iptables (filter/nat) ####"
    if have iptables; then iptables -S 2>/dev/null; echo "--- nat ---"; iptables -t nat -S 2>/dev/null; else echo "iptables NOT FOUND"; fi
    echo; echo "#### firewalld ####"
    if have firewall-cmd; then firewall-cmd --state 2>/dev/null; firewall-cmd --list-all 2>/dev/null; else echo "firewall-cmd NOT FOUND"; fi
'
# 自动判定
if have ss; then
    ss -tln 2>/dev/null | grep -q ":${MC_MASTER_PORT}\b" \
        && check "本机 ${MC_MASTER_PORT} mooncake-master 监听" 0 "本机在监听 master 端口" \
        || check "本机 ${MC_MASTER_PORT} mooncake-master 监听" 1 "本机未监听 ${MC_MASTER_PORT}(若 master 在本机则为异常)"
    ss -tln 2>/dev/null | grep -q ":${MC_META_PORT}\b" \
        && check "本机 ${MC_META_PORT} metadata 监听" 0 "本机在监听 metadata 端口" \
        || check "本机 ${MC_META_PORT} metadata 监听" 1 "本机未监听 ${MC_META_PORT}"
fi

# ============================================================================
# 03 RDMA 硬件: ibstat / ibv_devinfo / PCI
# ============================================================================
section "03_rdma_hw" bash -c "
    echo '#### lspci | Mellanox/Mellanox ####'
    if have lspci; then lspci 2>/dev/null | grep -iE 'mellanox|infiniband|connectx|ethernet' || echo '(无)'; else echo 'lspci NOT FOUND'; fi
    echo; echo '#### ibstat ####'
    if have ibstat; then ibstat 2>&1; else echo 'ibstat NOT FOUND(装 infiniband-diags)'; fi
    echo; echo '#### ibstatus ####'
    if have ibstatus; then ibstatus 2>&1; else echo 'ibstatus NOT FOUND'; fi
    echo; echo '#### ibv_devinfo ####'
    if have ibv_devinfo; then ibv_devinfo 2>&1; else echo 'ibv_devinfo NOT FOUND(装 libibverbs-utils/rdma-core)'; fi
    echo; echo '#### /sys/class/infiniband ####'
    ls -l /sys/class/infiniband/ 2>&1
    [ -r /sys/class/infiniband/${IB_DEV}/ports/1/state ] && echo \"state: \$(cat /sys/class/infiniband/${IB_DEV}/ports/1/state)\"
    [ -r /sys/class/infiniband/${IB_DEV}/ports/1/rate  ] && echo \"rate : \$(cat /sys/class/infiniband/${IB_DEV}/ports/1/rate)\"
"
# 自动判定
if have ibstat; then
    if ibstat 2>/dev/null | grep -q "^CA '${IB_DEV}'"; then
        check "RDMA 设备 ${IB_DEV} 存在" 0 "ibstat 可见 ${IB_DEV}"
        if ibstat 2>/dev/null | sed -n "/^CA '${IB_DEV}'/,/State:/p" | grep -qi 'Port state: Active'; then
            check "${IB_DEV} 端口状态" 0 "Port state: Active"
        else
            check "${IB_DEV} 端口状态" 1 "端口非 Active(物理链路/RDMA 没起来)"
        fi
    else
        check "RDMA 设备 ${IB_DEV} 存在" 1 "ibstat 未找到 ${IB_DEV}"
    fi
else
    check "RDMA 工具 ibstat" 2 "未安装 infiniband-diags, 无法判定"
fi

# ============================================================================
# 04 RDMA GID / PKey / LID  (mooncake RDMA 选路依赖)
# ============================================================================
section "04_rdma_gid" bash -c "
    echo '#### show_gids ####'
    if have show_gids; then show_gids 2>&1; else echo 'show_gids NOT FOUND'; fi
    echo; echo '#### /sys/class/infiniband/${IB_DEV}/ports/1/gids ####'
    for g in /sys/class/infiniband/${IB_DEV}/ports/*/gids/*; do [ -r \"\$g\" ] && echo \"\$g = \$(cat \"\$g\")\"; done 2>/dev/null
    echo; echo '#### pkeys ####'
    for p in /sys/class/infiniband/${IB_DEV}/ports/*/pkeys/*; do [ -r \"\$p\" ] && echo \"\$p = \$(cat \"\$p\")\"; done 2>/dev/null
    echo; echo '#### LID ####'
    for l in /sys/class/infiniband/${IB_DEV}/ports/*/lid; do [ -r \"\$l\" ] && echo \"\$l = \$(cat \"\$l\")\"; done 2>/dev/null
    echo; echo '#### ibv_query_port on ${IB_DEV} ####'
    if have ibv_devinfo; then ibv_devinfo -d ${IB_DEV} -v 2>&1 | sed -n '1,60p'; fi
"

# ============================================================================
# 05 RDMA 端口计数器 / 错误统计  (传输失败会在这里体现)
# ============================================================================
section "05_rdma_counters" bash -c "
    echo '#### /sys/class/infiniband/${IB_DEV}/ports/1/counters ####'
    for c in /sys/class/infiniband/${IB_DEV}/ports/*/counters/*; do
        [ -r \"\$c\" ] && printf '%s = %s\n' \"\${c##*counters/}\" \"\$(cat \"\$c\")\"
    done 2>/dev/null
    echo; echo '#### perfquery (若可用) ####'
    if have perfquery; then perfquery 2>&1 | head -60; else echo 'perfquery NOT FOUND'; fi
"
# 非零错误统计汇总
ERRS=""
for c in /sys/class/infiniband/${IB_DEV}/ports/*/counters/*; do
    [ -r "$c" ] || continue
    case "${c##*counters/}" in
        *symbol_error*|*link_error*|*port_*error*|*excessive_buffer*|*local_length_error*|*out_of_buffer*|*rpc_rdma_errors*)
            v="$(cat "$c" 2>/dev/null)"
            [ "${v:-0}" -gt 0 ] 2>/dev/null && ERRS="${ERRS} ${c##*counters/}=$v"
            ;;
    esac
done
[ -n "$ERRS" ] && check "RDMA 端口错误计数器(非零)" 1 "发现非零错误:$ERRS" || check "RDMA 端口错误计数器" 0 "关键错误计数为 0"

# ============================================================================
# 06 IB 内核模块 / 驱动 / dmesg
# ============================================================================
section "06_ib_drivers" bash -c '
    echo "#### lsmod | rdma/ib/mlx ####"; lsmod 2>/dev/null | grep -iE "ib_|mlx|rdma|nvme.*ofa" || echo "(无)"
    echo; echo "#### modinfo mlx5_core ####"; have modinfo && modinfo mlx5_core 2>/dev/null | grep -E "^(filename|version|srcversion)" || echo "modinfo NOT FOUND"
    echo; echo "#### rdma link / resource ####"
    if have rdma; then rdma link 2>&1; rdma resource 2>&1 | head -40; else echo "rdma NOT FOUND(iproute-rdma)"; fi
    echo; echo "#### /dev/infiniband ####"; ls -l /dev/infiniband/ 2>&1
    echo; echo "#### dmesg | mlx/ib/rdma (近 500 行) ####"
    if have dmesg; then dmesg 2>/dev/null | grep -iE "mlx|infiniband|ib_|rdma|mooncake" | tail -60 || echo "(无匹配)"; else echo "dmesg 不可读(需 root)"; fi
'
# uverbs 设备可访问性
[ -e "/dev/infiniband/uverbs0" ] \
    && { [ -r "/dev/infiniband/uverbs0" ] && check "/dev/infiniband/uverbs0 可读" 0 "存在且可读" || check "/dev/infiniband/uverbs0 可读" 1 "存在但当前用户不可读(需 root/rdma 组)"; } \
    || check "/dev/infiniband/uverbs0 存在" 1 "不存在, RDMA 未就绪"

# ============================================================================
# 07 mooncake 进程 / 端口 / 日志 / 配置
# ============================================================================
section "07_mooncake" bash -c "
    echo '#### 进程 (mooncake/mc_) ####'
    ps -ef 2>/dev/null | grep -iE 'mooncake|mc_master|mc_daemon|metadata' | grep -vE 'grep|diag_mooncake' || echo '(无 mooncake 相关进程)'
    echo; echo '#### mooncake 监听端口 (按 pid 归属) ####'
    if have ss; then ss -tlnp 2>/dev/null | grep -iE 'mooncake|50051|8080' || echo '(ss 未见)'; fi
    echo; echo '#### mooncake 相关文件 ####'
    ls -ld /etc/mooncake* /opt/mooncake* /var/log/mooncake* 2>/dev/null || echo '(无常见路径)'
    find / -maxdepth 4 -iname 'mooncake*.json' -o -iname 'mooncake*.conf' 2>/dev/null | head -20
    echo; echo '#### 本机 master 端口 50051 自连 ####'
    if have curl; then curl -m 3 -sS \"http://127.0.0.1:${MC_MASTER_PORT}/\" 2>&1 | head -5 || echo '(curl 失败/无 HTTP 服务)'; else echo 'curl NOT FOUND'; fi
    echo; echo '#### TE metadata 自查 ####'
    if have curl; then curl -m 3 -sS \"http://127.0.0.1:${MC_META_PORT}/metadata\" 2>&1 | head -20 || echo '(metadata 不可达)'; else echo 'curl NOT FOUND'; fi
    echo; echo '#### journalctl mooncake (近 200 行) ####'
    if have journalctl; then journalctl -n 400 --no-pager 2>/dev/null | grep -iE 'mooncake|mc_' | tail -80 || echo '(无匹配/无权限)'; else echo 'journalctl NOT FOUND'; fi
    echo; echo '#### /tmp 下 mooncake 日志 ####'
    ls -lt /tmp/*mooncake* /tmp/mooncake* 2>/dev/null | head || echo '(无)'
"
# mooncake 进程判定
PSC="$(ps -ef 2>/dev/null | grep -iE 'mooncake|mc_master|mc_daemon' | grep -vcE 'grep|diag_mooncake')"
[ "${PSC:-0}" -gt 0 ] \
    && check "mooncake 相关进程" 0 "发现 ${PSC} 个 mooncake 相关进程" \
    || check "mooncake 相关进程" 2 "未发现独立 mooncake 进程(若为 sglang in-process 嵌入则正常, 否则异常)"

# ============================================================================
# 08 Docker 容器 (含已停容器): 看 IB 挂载 / devices / caps / env / ulimits
#    注: 出问题后手动 stop, 容器可能 stop 未 rm -> inspect 仍可用; exec 需 running
# ============================================================================
section "08_docker" bash -c "
    if ! have docker; then echo 'docker NOT FOUND(或当前用户无 docker 权限)'; exit 0; fi
    echo '#### docker ps -a (相关容器) ####'
    docker ps -a --format 'table {{.ID}}\t{{.Image}}\t{{.Status}}\t{{.Names}}' 2>&1 | head -40
    echo; echo '#### 定位本次 PD 容器 ####'
    CIDS=\$(docker ps -a --format '{{.ID}} {{.Names}}' 2>/dev/null | grep -iE '${CONTAINER_PAT}|mooncake|sglang|router' | awk '{print \$1}')
    echo \"命中容器: \${CIDS:-<none>}\"
    for cid in \$CIDS; do
        echo; echo \"==================== container \$cid ====================\"
        echo '--- 名称/状态 ---'; docker ps -a --format '{{.Names}} | {{.Status}} | {{.Image}}' --filter id=\$cid 2>&1
        echo '--- HostConfig.Devices (期望含 /dev/infiniband) ---'
        docker inspect \$cid --format '{{json .HostConfig.Devices}}' 2>&1
        echo '--- CapAdd (期望含 IPC_LOCK) ---'
        docker inspect \$cid --format '{{json .HostConfig.CapAdd}}' 2>&1
        echo '--- Ulimits (期望 memlock=-1:-1) ---'
        docker inspect \$cid --format '{{json .HostConfig.Ulimits}}' 2>&1
        echo '--- Env (MOONCAKE_*) ---'
        docker inspect \$cid --format '{{range .Config.Env}}{{println .}}{{end}}' 2>&1 | grep -iE 'MOONCAKE|DISAGG|RDMA|IB' || echo '(无)'
        echo '--- Mounts (infiniband/host ipc) ---'
        docker inspect \$cid --format '{{json .Mounts}}' 2>&1
        echo '--- 网络模式 ---'
        docker inspect \$cid --format '{{.HostConfig.NetworkMode}}' 2>&1
        echo '--- 若 running, 容器内 RDMA 视图 ---'
        if docker inspect \$cid --format '{{.State.Running}}' 2>/dev/null | grep -q true; then
            docker exec \$cid bash -lc 'ibstat 2>/dev/null; echo ---; ls -l /dev/infiniband/ 2>/dev/null; echo ---; cat /proc/self/limits 2>/dev/null | grep -iE \"memlock|nofile\"' 2>&1 | head -40
        else
            echo '(容器未运行, 跳过 exec; 请在重新拉起后补采容器内 ibstat)'
        fi
    done
"

# ============================================================================
# 09 ulimit / 内存锁定 / hugepages  (mooncake 注册 RDMA segment 必需)
# ============================================================================
section "09_ulimit_mem" bash -c '
    echo "#### 当前 shell limits ####"; bash -lc "ulimit -a" 2>&1
    echo; echo "#### memlock ####"; bash -lc "ulimit -l" 2>&1
    echo; echo "#### /proc/meminfo ####"; grep -E "MemTotal|MemFree|MemAvailable|Huge" /proc/meminfo
    echo; echo "#### hugepages ####"; cat /proc/sys/vm/nr_hugepages 2>/dev/null; ls -d /sys/kernel/mm/hugepages/* 2>/dev/null
    echo; echo "#### 锁定内存占用 ####"; grep -iE "locked|rss" /proc/*/status 2>/dev/null | head -0; awk "/VmLck/{s+=\$2} END{print \"total VmLck(KB): \" s}" /proc/*/status 2>/dev/null
'
ML="$(bash -lc 'ulimit -l' 2>/dev/null)"
case "${ML:-}" in
    unlimited) check "memlock ulimit (-l)" 0 "unlimited";;
    ''|*[!0-9]*) check "memlock ulimit (-l)" 2 "无法判定 ($ML)";;
    *) if [ "$ML" -ge 65536 ] 2>/dev/null; then check "memlock ulimit (-l)" 0 "$ML KB"; else check "memlock ulimit (-l)" 1 "$ML KB(过小, mooncake 注册大 RDMA segment 会失败, 应设 unlimited)"; fi;;
esac

# ============================================================================
# 10 到对端的连通性 (TCP 控制面 + RDMA 数据面探测)
# ============================================================================
section "10_peer_conn" bash -c "
    echo \"#### 对端: \$1 (\$2) ####\"
    [ -z \"\$1\" ] && { echo '未识别对端(本机 IP 既非 13 也非 14), 跳过'; exit 0; }
    for svc in master:${MC_MASTER_PORT} meta:${MC_META_PORT}; do
        name=\${svc%%:*}; port=\${svc##*:}
        printf 'TCP %-8s %s:%s ... ' \"\$name\" \"\$1\" \"\$port\"
        if timeout 4 bash -c \"exec 3<>/dev/tcp/\$1/\$port\" 2>/dev/null; then echo OK; else echo FAIL; fi
    done
    echo; echo '#### mooncake 历史失败数据端口示例 (仅供对照, 现场多为动态) ####'
    for port in ${MC_PORT_HINTS}; do
        printf 'TCP hint %s:%s ... ' \"\$1\" \"\$port\"
        if timeout 3 bash -c \"exec 3<>/dev/tcp/\$1/\$port\" 2>/dev/null; then echo OK; else echo \"FAIL/closed(运行期才有)\"; fi
    done
    echo; echo '#### RDMA 数据面建议双向测试命令 (需对端起 server) ####'
    echo \"  对端先起: ib_write_bw -d ${IB_DEV} --report_gbits\"
    echo \"  本机再打: ib_write_bw -d ${IB_DEV} \$1 --report_gbits\"
    echo \"  若无 perftest, 用: rdma_ping / ibping(在 infiniband-diags)\"
    have ib_write_bw && echo '(本机已安装 ib_write_bw)' || echo '(本机未安装 perftest, 建议 yum install -y perftest)'
" _ "$PEER_IP"
[ -n "$PEER_IP" ] && timeout 4 bash -c "exec 3<>/dev/tcp/${PEER_IP}/${MC_MASTER_PORT}" 2>/dev/null \
    && check "对端 ${PEER_IP}:${MC_MASTER_PORT} (master) TCP 连通" 0 "可达" \
    || check "对端 ${PEER_IP}:${MC_MASTER_PORT} (master) TCP 连通" $([ -z "$PEER_IP" ] && echo 2 || echo 1) "$([ -z "$PEER_IP" ] && echo '未识别对端' || echo '不可达')"

# ============================================================================
# 11 nvidia-smi (KV cache 所在, 顺便确认 GPU 健康)
# ============================================================================
section "11_gpu" bash -c '
    if have nvidia-smi; then nvidia-smi 2>&1 | head -40; else echo "nvidia-smi NOT FOUND"; fi
'

# ============================================================================
# 12 收尾: 打包提示
# ============================================================================
{
    echo
    echo "------ 各分类明细见同目录 *.txt ------"
    echo "01_system 02_network 03_rdma_hw 04_rdma_gid 05_rdma_counters"
    echo "06_ib_drivers 07_mooncake 08_docker 09_ulimit_mem 10_peer_conn 11_gpu"
    echo
    echo "------ 解读提示 ------"
    echo "* decode(14) 侧: 若 ${MC_MASTER_PORT}/${MC_META_PORT} 未监听 或 无 mooncake 进程"
    echo "  -> 对应错误 'Decode instance could be dead, remote mooncake session'"
    echo "* 若 mlx5_0 端口非 Active / 有非零错误计数 / uverbs 不可读"
    echo "  -> 对应 'Failed to send kv chunk'(RDMA 数据面不通)"
    echo "* 若容器未挂 /dev/infiniband 或 memlock 非 unlimited"
    echo "  -> mooncake 注册 RDMA segment 失败"
    echo "* 强烈建议: 同一脚本在 prefill(13) 上也跑一份做对照, 重点比对"
    echo "  03/04(端口Active+GID) 与 10(双向连通)."
    echo
    echo "建议双向 RDMA 带宽实测: ib_write_bw -d ${IB_DEV} <对端IP> --report_gbits"
} >> "$SUMMARY"

echo
echo "==================== 完成 ===================="
echo "结果目录: $OUTDIR"
echo "快速判定: $SUMMARY"
echo
cat "$SUMMARY"
echo
echo "（如需上报, 把整个 $OUTDIR 目录打包即可）"
exit 0
