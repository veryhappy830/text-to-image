module.exports = {
  apps: [{
    name: 't2i-server',
    script: 'serve.py',
    args: '',
    interpreter: '/venv/text-to-image/bin/python',
    log_file: 'logs/mvdream-server.log',
    out_file: 'logs/mvdream-out.log',
    error_file: 'logs/mvdream-error.log',
    time: true,
    merge_logs: true,
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '8G',
    wait_ready: true,
    listen_timeout: 10000,
    kill_timeout: 5000,
  }]
};