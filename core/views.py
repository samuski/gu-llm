# core/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from core.llm.inference import generate

@csrf_exempt
def chat(request):
    if request.method != "POST":
        return JsonResponse({"detail": "POST required"}, status=405)
    data = json.loads(request.body or "{}")
    messages = data.get("messages", [])
    max_new = int(data.get("max_new_tokens", 256))
    print("[chat] max_new_tokens from client:", max_new)
    text = generate(messages=messages, max_new_tokens=max_new)
    return JsonResponse({"response": text})
