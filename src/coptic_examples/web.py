"""Local dictionary-compatible research UI and stable JSON search endpoint."""
import json
from http.server import BaseHTTPRequestHandler,HTTPServer
from urllib.parse import parse_qs,urlsplit
from .retrieval import Search
from .ui import search_page,guide_page,experiments_page


def serve(config,port=8012):
    search=Search(config)
    reports={'ABLATIONS.md','LEGACY_REVIEW.md','ablation_results.json','barley_loo_diagnostic.json','CRUM_BIBLE_AUDIT.md'}

    class Handler(BaseHTTPRequestHandler):
        def send(self,status,kind,data):
            if isinstance(data,str):data=data.encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(data)))
            self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers();self.wfile.write(data)

        def error(self,exc,api=False):
            self.send(400,'application/json; charset=utf-8' if api else 'text/html; charset=utf-8',
                json.dumps(dict(error=str(exc))) if api else search_page(config,error=str(exc)))

        def find(self,fields,default='research'):
            ranker=str(fields.get('ranker',default))
            if ranker not in {'research','cheap','learned','quality_first','linear'}:raise ValueError('Unknown ranker')
            return search.search(str(fields.get('query','')),ranker=ranker,dialect=str(fields.get('dialect','sahidic')))

        def do_GET(self):
            route=urlsplit(self.path);path=route.path
            fields={k:v[0] for k,v in parse_qs(route.query).items()}
            try:
                if path=='/font.woff':return self.send(200,'font/woff',(config.source/'dictionary/public/antinoou-webfont.woff').read_bytes())
                if path=='/review.css':return self.send(200,'text/css; charset=utf-8',(config.work/'assets/review.css').read_bytes())
                if path=='/guide':return self.send(200,'text/html; charset=utf-8',guide_page(config))
                if path=='/experiments':return self.send(200,'text/html; charset=utf-8',experiments_page(config,fields.get('dialect','sahidic'),fields.get('task','acceptance')))
                if path.startswith('/reports/'):
                    name=path.removeprefix('/reports/')
                    if name not in reports:return self.send(404,'text/plain','Not found')
                    return self.send(200,'application/json; charset=utf-8' if name.endswith('.json') else 'text/plain; charset=utf-8',(config.reports/name).read_bytes())
                if path not in {'/','/search'}:return self.send(404,'text/plain','Not found')
                result=self.find(fields) if fields.get('query') else None
                return self.send(200,'text/html; charset=utf-8',search_page(config,result))
            except (ValueError,KeyError,FileNotFoundError,RuntimeError) as exc:self.error(exc)

        def do_POST(self):
            path=urlsplit(self.path).path;api=path=='/api/search'
            if path not in {'/search','/api/search'}:return self.send(404,'text/plain','Not found')
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=65536:raise ValueError('Invalid request length')
                raw=self.rfile.read(length).decode('utf-8')
                fields=json.loads(raw) if api else {k:v[0] for k,v in parse_qs(raw).items()}
                if not isinstance(fields,dict):raise ValueError('Expected a JSON object')
                result=self.find(fields,'cheap' if api else 'research')
                return self.send(200,'application/json; charset=utf-8' if api else 'text/html; charset=utf-8',
                    json.dumps(result,ensure_ascii=False) if api else search_page(config,result))
            except (ValueError,KeyError,FileNotFoundError,RuntimeError) as exc:self.error(exc,api)

    server=HTTPServer(('127.0.0.1',port),Handler)
    print(f'Open http://localhost:{port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
