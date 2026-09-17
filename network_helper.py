"""
Adaab Studio - Network and Mobile Access Helper
Handles local IP detection, self-signed SSL certificates for mobile microphone access,
and QR code generation for quick smartphone onboarding.
"""

import os
import sys
import socket
import datetime
import ipaddress
import io
import base64

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"

def get_local_lan_ip() -> str:
    """Detects the PC's local network IP address (e.g. 192.168.1.2)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def ensure_self_signed_cert(cert_dir: str = None, host_ip: str = None) -> tuple[str, str]:
    """
    Generates a self-signed SSL certificate and private key.
    Enables HTTPS over the local network so mobile browsers (iOS Safari & Android Chrome)
    grant microphone (navigator.mediaDevices.getUserMedia) permissions.
    """
    cert_dir = cert_dir or os.path.dirname(os.path.abspath(__file__))
    cert_path = os.path.join(cert_dir, "adaab_cert.pem")
    key_path = os.path.join(cert_dir, "adaab_key.pem")

    ip = host_ip or get_local_lan_ip()

    if os.path.exists(cert_path) and os.path.exists(key_path):
        return cert_path, key_path

    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "PK"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Adaab Studio"),
        x509.NameAttribute(NameOID.COMMON_NAME, ip),
    ])

    san_list = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1"))
    ]
    try:
        san_list.append(x509.IPAddress(ipaddress.IPv4Address(ip)))
    except Exception:
        pass

    san = x509.SubjectAlternativeName(san_list)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
        .add_extension(san, critical=False)
        .sign(key, hashes.SHA256())
    )

    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))

    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    return cert_path, key_path

def generate_qr_code_ascii(url: str) -> str:
    """Generates an ASCII text QR code for rendering directly in Windows terminal."""
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        out = io.StringIO()
        qr.print_ascii(out=out, invert=True)
        return out.getvalue()
    except Exception:
        return ""

def generate_qr_code_base64(url: str) -> str:
    """Generates a PNG base64 data URI of the QR code for web display."""
    try:
        import qrcode
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=6,
            border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="#000000", back_color="#ffffff")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        print(f"[NetworkHelper] QR code generation notice: {e}")
        return ""

def is_port_in_use(port: int, host: str = "0.0.0.0") -> bool:
    """Checks if a TCP port is currently in use or cannot be bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True

def terminate_all_stale_app_processes():
    """Kills any stale python process running app.py on Adaab ports (7865-7885) and stale cloudflared. Never touches port 7860."""
    if sys.platform != "win32":
        return
    import subprocess
    try:
        subprocess.run("taskkill /F /IM cloudflared.exe", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    curr_pid = os.getpid()
    for port in range(7865, 7885):
        if port <= 7864:
            continue  # Explicitly preserve 7860 and any external service
        try:
            output = subprocess.check_output(f'netstat -ano | findstr ":{port}"', shell=True, text=True, stderr=subprocess.DEVNULL)
            for line in output.strip().splitlines():
                parts = line.split()
                if len(parts) >= 5 and "LISTENING" in parts:
                    pid = int(parts[-1])
                    if pid != curr_pid and pid > 0:
                        ps_cmd = f"(Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}').CommandLine"
                        cmd_line = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps_cmd], text=True, stderr=subprocess.DEVNULL).strip()
                        if "app.py" in cmd_line:
                            print(f"[Adaab] Cleaned up previous stale app.py (PID {pid} on port {port})...", flush=True)
                            subprocess.run(f"taskkill /F /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

def resolve_available_port(requested_port: int = 7865, auto_free_stale: bool = True) -> int:
    """
    Ensures a valid, non-conflicting port is available:
    1. Strictly never uses or touches port 7860 (defaults to 7865+).
    2. Checks if the requested port is free. If so, returns it.
    3. If occupied on Windows and auto_free_stale is True:
       Checks if the process using the port is an old/stale instance of Python running 'app.py'.
       If so, terminates the stale instance and reclaims the requested port.
    4. If the port remains occupied (by an external service or permission limit),
       scans upwards for the next free port (e.g. 7866, 7867, ...) so startup never crashes.
    """
    if requested_port <= 7864:
        requested_port = 7865  # Never touch 7860

    if not is_port_in_use(requested_port):
        return requested_port

    import subprocess
    if auto_free_stale and sys.platform == "win32":
        try:
            # Query netstat for PID listening on requested_port
            output = subprocess.check_output(f'netstat -ano | findstr ":{requested_port}"', shell=True, text=True)
            for line in output.strip().splitlines():
                parts = line.split()
                if len(parts) >= 5 and "LISTENING" in parts:
                    pid = int(parts[-1])
                    if pid != os.getpid() and pid > 0:
                        ps_cmd = f"(Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}').CommandLine"
                        try:
                            cmd_line = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps_cmd], text=True, stderr=subprocess.DEVNULL).strip()
                            if "app.py" in cmd_line:
                                print(f"[Adaab] Port {requested_port} is held by previous stale app.py (PID {pid}). Freeing port...", flush=True)
                                subprocess.run(f"taskkill /F /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                import time
                                time.sleep(0.8)
                                if not is_port_in_use(requested_port):
                                    print(f"[Adaab] Successfully reclaimed port {requested_port}!", flush=True)
                                    return requested_port
                        except Exception:
                            pass
        except Exception:
            pass

    # If still occupied, search upwards for next available port starting from max(7865, requested_port + 1)
    start_search = max(7865, requested_port + 1)
    for candidate in range(start_search, start_search + 30):
        if not is_port_in_use(candidate):
            print(f"[Adaab] Notice: Port {requested_port} is busy. Automatically assigned free port {candidate}.", flush=True)
            return candidate

    return requested_port

def ensure_cloudflared() -> str:
    """Ensures cloudflared executable is present locally in the project directory."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    cf_exe = os.path.join(project_dir, "cloudflared.exe")
    if os.path.exists(cf_exe):
        return cf_exe
    import urllib.request
    print("[Adaab] Downloading Cloudflare Quick Tunnel binary (firewall-proof)...", flush=True)
    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    urllib.request.urlretrieve(url, cf_exe)
    return cf_exe

def is_valid_cloudflare_tunnel_url(url: str) -> bool:
    """
    Verifies whether a URL is a legitimate user-facing Cloudflare Quick Tunnel URL.
    Explicitly filters out Cloudflare internal API endpoints (e.g. api.trycloudflare.com)
    and administrative subdomains.
    """
    if not url or not isinstance(url, str):
        return False
    import re
    cleaned = url.strip().lower()
    match = re.match(r"^https://([a-zA-Z0-9-]+)\.trycloudflare\.com/?$", cleaned)
    if not match:
        return False
    subdomain = match.group(1).lower()
    # Explicitly blacklisted internal / administrative service subdomains
    if subdomain in ("api", "update", "dash", "developers", "www", "blog", "status", "test"):
        return False
    # Quick tunnel subdomains are random slug names (e.g. "copper-closer-keys")
    return len(subdomain) >= 4

def start_cloudflared_tunnel(local_port: int, timeout_sec: int = 35) -> tuple[str | None, any]:
    """
    Spins up a Cloudflare Quick Tunnel pointing to http://127.0.0.1:{local_port}.
    Bypasses firewalls and provides a verified Let's Encrypt / Cloudflare HTTPS link.
    Redirects output to cloudflared.log to prevent Windows pipe deadlocks.
    Returns (public_https_url, process) or (None, None).
    """
    import subprocess
    import re
    import time
    cf_exe = ensure_cloudflared()
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloudflared.log")
    log_file = open(log_path, "w", encoding="utf-8", errors="replace")
    
    cmd = [
        cf_exe, "tunnel",
        "--url", f"http://127.0.0.1:{local_port}",
        "--http-host-header", f"127.0.0.1:{local_port}",
        "--no-autoupdate"
    ]
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    
    start_t = time.time()
    tunnel_url = None
    while time.time() - start_t < timeout_sec:
        time.sleep(0.5)
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                    matches = re.findall(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content, re.IGNORECASE)
                    for cand in matches:
                        cand_clean = cand.strip().lower()
                        if is_valid_cloudflare_tunnel_url(cand_clean):
                            return cand_clean, proc
            except Exception:
                pass
        if proc.poll() is not None:
            break
    return tunnel_url, proc



def is_secure_context_origin(url: str) -> bool:
    """
    Validates whether a URL meets the W3C Secure Context specification
    required by mobile browsers to enable navigator.mediaDevices.getUserMedia.
    Rules:
    - https:// origins are always secure contexts.
    - localhost / 127.0.0.1 are secure contexts even over plain http://.
    - Plain http:// on non-loopback IP (e.g. 192.168.x.x) is NOT a secure context.
    """
    if not url:
        return False
    lower = url.lower().strip()
    if lower.startswith("https://"):
        return True
    if lower.startswith("http://localhost") or lower.startswith("http://127.0.0.1"):
        return True
    return False

def start_localtunnel(local_port: int, subdomain: str = "adaab-studio", timeout_sec: int = 25) -> tuple[str | None, any]:
    """
    Spins up a Localtunnel pointing to http://localhost:{local_port} with an appropriate branded subdomain.
    Produces clean URLs like https://adaab-studio.loca.lt
    Returns (tunnel_url, process) or (None, None).
    """
    import subprocess
    import re
    import time
    cmd = ["cmd", "/c", f"npx localtunnel --port {local_port} --subdomain {subdomain}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    
    start_t = time.time()
    tunnel_url = None
    while time.time() - start_t < timeout_sec:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            match = re.search(r"https://[a-zA-Z0-9-]+\.loca\.lt", line)
            if match:
                tunnel_url = match.group(0)
                break
    return tunnel_url, proc

if __name__ == "__main__":
    ip = get_local_lan_ip()
    print(f"Local LAN IP: {ip}")
    cert, key = ensure_self_signed_cert(host_ip=ip)
    print(f"SSL Cert: {cert}\nSSL Key: {key}")
    url = f"https://{ip}:7865"
    print(f"\nMobile URL: {url}\n")
    print(generate_qr_code_ascii(url))