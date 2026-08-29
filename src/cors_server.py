from http.server import HTTPServer, SimpleHTTPRequestHandler
import sys
import os
from pathlib import Path

# Xác định thư mục gốc dự án (thư mục cha của src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super(CORSRequestHandler, self).end_headers()

if __name__ == '__main__':
    # Chuyển working directory về thư mục gốc dự án để serve file đúng đường dẫn
    os.chdir(PROJECT_ROOT)
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
    print(f"Starting CORS HTTP server on port {port}")
    print(f"Serving files from: {PROJECT_ROOT}")
    httpd = HTTPServer(('localhost', port), CORSRequestHandler)
    httpd.serve_forever()
