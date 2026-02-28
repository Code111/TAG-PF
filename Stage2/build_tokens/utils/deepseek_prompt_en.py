# # # utils/deepseek_prompt_en.py
# # import os
# # import json
# # import hashlib
# # from typing import Any, Dict, Optional

# # from openai import OpenAI

# # class DeepSeekPromptEN:
# #     """
# #     Call DeepSeek (OpenAI-compatible) Chat API to generate a dynamic English prompt_text
# #     from one window summary (JSON).
# #     """

# #     def __init__(
# #         self,
# #         api_key: Optional[str] = None,
# #         base_url: str = "https://api.deepseek.com",
# #         model: str = "deepseek-chat",
# #         timeout: int = 60,
# #         temperature: float = 0.2,
# #         max_tokens: int = 400,
# #         enable_cache: bool = True,
# #     ):
# #         self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
# #         if not self.api_key:
# #             raise RuntimeError(
# #                 "DeepSeek API key missing. Set env DEEPSEEK_API_KEY or pass api_key=..."
# #             )

# #         self.client = OpenAI(api_key=self.api_key, base_url=base_url, timeout=timeout)
# #         self.model = model
# #         self.temperature = temperature
# #         self.max_tokens = max_tokens
# #         self.enable_cache = enable_cache
# #         self._cache: Dict[str, str] = {}

# #         # ✅ final system prompt (你确认的版本，稍微做了工程性强化：强调“指令式”“干货”“不复述表”)
# #         self.system = (
# #             "You are an assistant that writes an English prompt_text to CONDITION a PV power forecasting model.\n"
# #             "The model already receives the full numeric time-series; this prompt_text provides compact, "
# #             "data-driven guidance on how the historical window should be interpreted for forecasting.\n"
# #             "You will receive a JSON summary of ONE historical window sampled every 15 minutes (96 steps, 24 hours).\n\n"
# #             "The historical data includes: date, Total solar irradiance (W/m2), "
# #             "Direct normal irradiance (W/m2), Global horizontal irradiance (W/m2), and OT (MW).\n\n"
# #             "Output MUST be strict JSON only: {\"prompt_text\":\"...\"}\n"
# #             "Use \\n exactly once per line break inside the JSON string.\n"
# #             "Do NOT output anything outside the JSON.\n\n"
# #             "Write prompt_text as EXACTLY 5 lines with these fixed prefixes. "
# #             "Each line MUST start with the exact prefix shown below:\n"
# #             "1) PV power forecasting task.\n"
# #             "2) Window: <start>–<end>.\n"
# #             "3) Night/Day: describe the diurnal structure as operational rules, "
# #             "explicitly stating BOTH (a) a night regime with near-zero irradiance and OT, "
# #             "and (b) a daytime regime with active generation and rise–peak–fall behavior. "
# #             "Use ONLY provided times or ranges if mentioned; otherwise describe qualitatively.\n"
# #             "4) Pattern: describe the weather-driven and diurnal behavior in meteorologist-style language, "
# #             "focusing on stability vs variability, smoothness vs fluctuations, and persistence. "
# #             "Translate the observed behavior into guidance for the model "
# #             "(e.g., treat the profile as smooth and persistent vs potentially volatile). "
# #             "Avoid vague adjectives unless supported by observed data behavior.\n"
# #             "5) Goal: use historical irradiance and historical OT to forecast OT for the next 24 hours.\n\n"
# #             "Rules:\n"
# #             "- Use ONLY information supported by the provided data; never invent values.\n"
# #             "- Do NOT restate the full numeric table or variable definitions.\n"
# #             "- Do NOT specify explicit future peak values or future timestamps.\n"
# #             "- Keep the text concise and information-dense (<= ~1000 characters total).\n"
# #         )

# #     def _cache_key(self, window_summary: Dict[str, Any]) -> str:
# #         s = json.dumps(window_summary, ensure_ascii=False, sort_keys=True)
# #         return hashlib.md5(s.encode("utf-8")).hexdigest()

# #     def generate_prompt_text(self, window_summary: Dict[str, Any]) -> str:
# #         """
# #         Returns prompt_text (string with real newlines).
# #         The LLM output must be JSON: {"prompt_text":"line1\nline2..."}
# #         """
# #         key = self._cache_key(window_summary)
# #         if self.enable_cache and key in self._cache:
# #             return self._cache[key]

# #         user_msg = json.dumps(window_summary, ensure_ascii=False)
        
# #         resp = self.client.chat.completions.create(
# #             model=self.model,
# #             messages=[
# #                 {"role": "system", "content": self.system},
# #                 {"role": "user", "content": user_msg},
# #             ],
# #             temperature=self.temperature,
# #             max_tokens=self.max_tokens,
# #             stream=False,
# #         )

# #         content = resp.choices[0].message.content.strip()

# #         # Parse strict JSON
# #         try:
# #             obj = json.loads(content)
# #             prompt_text = obj["prompt_text"]
# #             if not isinstance(prompt_text, str):
# #                 raise ValueError("prompt_text is not a string")
# #         except Exception as e:
# #             raise RuntimeError(f"DeepSeek returned non-JSON or invalid JSON. Raw:\n{content}\nError: {e}")

# #         if self.enable_cache:
# #             self._cache[key] = prompt_text
# #         return prompt_text
# # utils/deepseek_prompt_en.py
# import os
# import json
# import hashlib
# from typing import Any, Dict, Optional

# from openai import OpenAI


# class DeepSeekPromptEN:
#     """
#     Call LOCAL vLLM (OpenAI-compatible) to generate prompt_text.
#     Serial (no concurrency), with disk cache.
#     """

#     def __init__(
#         self,
#         api_key: Optional[str] = None,
#         base_url: str = "http://127.0.0.1:8000/v1",  # ✅ 改：本地 vLLM
#         model: str = "Llama-3.3-70B-Instruct",
#         timeout: int = 60,
#         temperature: float = 0.0,
#         max_tokens: int = 220,
#         enable_cache: bool = True,
#         cache_dir: str = "./prompt_cache",          # ✅ 改：磁盘 cache
#     ):
#         # vLLM 不校验 key
#         self.api_key = api_key or "EMPTY"
#         self.client = OpenAI(api_key=self.api_key, base_url=base_url, timeout=timeout)

#         self.model = model
#         self.temperature = temperature
#         self.max_tokens = max_tokens

#         self.enable_cache = enable_cache
#         self.cache_dir = cache_dir
#         if self.enable_cache:
#             os.makedirs(self.cache_dir, exist_ok=True)

#         # ===== system prompt（你原来的，完全保留）=====
#         self.system = (
#             "You are an assistant that writes an English prompt_text to CONDITION a PV power forecasting model.\n"
#             "The model already receives the full numeric time-series; this prompt_text provides compact, "
#             "data-driven guidance on how the historical window should be interpreted for forecasting.\n"
#             "You will receive a JSON summary of ONE historical window sampled every 15 minutes (96 steps, 24 hours).\n\n"
#             "The historical data includes: date, Total solar irradiance (W/m2), "
#             "Direct normal irradiance (W/m2), Global horizontal irradiance (W/m2), and OT (MW).\n\n"
#             "Output MUST be strict JSON only: {\"prompt_text\":\"...\"}\n"
#             "Use \\n exactly once per line break inside the JSON string.\n"
#             "Do NOT output anything outside the JSON.\n\n"
#             "Write prompt_text as EXACTLY 5 lines with these fixed prefixes. "
#             "Each line MUST start with the exact prefix shown below:\n"
#             "1) PV power forecasting task.\n"
#             "2) Window: <start>–<end>.\n"
#             "3) Night/Day: describe the diurnal structure as operational rules, "
#             "explicitly stating BOTH (a) a night regime with near-zero irradiance and OT, "
#             "and (b) a daytime regime with active generation and rise–peak–fall behavior. "
#             "Use ONLY provided times or ranges if mentioned; otherwise describe qualitatively.\n"
#             "4) Pattern: describe the weather-driven and diurnal behavior in meteorologist-style language, "
#             "focusing on stability vs variability, smoothness vs fluctuations, and persistence. "
#             "Translate the observed behavior into guidance for the model.\n"
#             "5) Goal: use historical irradiance and historical OT to forecast OT for the next 24 hours.\n"
#         )

#     # ---------- 磁盘 cache ----------
#     def _cache_key(self, window_summary: Dict[str, Any]) -> str:
#         s = json.dumps(window_summary, ensure_ascii=False, sort_keys=True)
#         return hashlib.md5(s.encode("utf-8")).hexdigest()

#     def _cache_path(self, key: str) -> str:
#         return os.path.join(self.cache_dir, f"{key}.json")

#     def generate_prompt_text(self, window_summary: Dict[str, Any]) -> str:
#         key = self._cache_key(window_summary)

#         # ✅ cache 命中：直接返回（毫秒级）
#         if self.enable_cache:
#             path = self._cache_path(key)
#             if os.path.exists(path):
#                 with open(path, "r", encoding="utf-8") as f:
#                     return json.load(f)["prompt_text"]

#         # ===== 串行调用 vLLM =====
#         user_msg = json.dumps(window_summary, ensure_ascii=False)

#         resp = self.client.chat.completions.create(
#             model=self.model,
#             messages=[
#                 {"role": "system", "content": self.system},
#                 {"role": "user", "content": user_msg},
#             ],
#             temperature=self.temperature,
#             max_tokens=self.max_tokens,
#             stream=False,
#         )

#         content = resp.choices[0].message.content.strip()
#         print(content)
#         obj = json.loads(content)
#         prompt_text = obj["prompt_text"]

#         # 写入磁盘 cache
#         if self.enable_cache:
#             with open(self._cache_path(key), "w", encoding="utf-8") as f:
#                 json.dump({"prompt_text": prompt_text}, f, ensure_ascii=False)

#         return prompt_text





# utils/deepseek_prompt_en.py
import json
import time
from typing import Any, Dict, Optional, List
from openai import OpenAI


class DeepSeekPromptEN:
    """
    规则：
      - 不做缓存
      - 不做“修补/抽取/兜底改写”
      - 只要输出不达标 => 重试
      - 达到重试上限 => 抛异常（由外层决定 fallback）
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "llama3-8b",
        timeout: int = 60,
        temperature: float = 0.0,
        max_tokens: int = 500,
        use_response_format: bool = False,
        max_retries: int = 24,
        retry_sleep: float = 0.2,
    ):
        self.client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_response_format = use_response_format
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep

        self.system = (
            "You are a wind power data feature extractor. Your task is to convert a JSON summary of ONE "
            "24-hour historical window (sampled every 15 minutes) into a concise, objective English prompt "
            "that conditions a downstream wind power forecasting model.\n\n"
            "The wind dataset contains: Time (year-month-day h:m:s), wind speed at 30 m (m/s), wind speed at 50 m (m/s), "
            "wind speed at hub height (m/s), and Power (MW).\n\n"
            "Output MUST be strict JSON only: {\"prompt_text\":\"...\"}. Do not include any content outside the JSON. "
            "Inside the JSON string, use \\n exactly once per line break.\n\n"
            "Write a focused 80–100 word prompt based ONLY on observed behavior in the input data. "
            "The prompt_text MUST include the following elements:\n"
            "1) Task: explicitly state that the goal is to forecast wind power (MW) for the next 24 hours.\n"
            "2) Time Window: report the exact start–end timestamps of the historical data window.\n"
            "3) Wind Regime Timing (NUMERIC REQUIRED when identifiable): identify sustained low-wind periods versus active-wind periods "
            "using wind speed at hub height and the corresponding power response. "
            "When clearly identifiable, report explicit time ranges in HH:MM–HH:MM using ONLY times observed in the historical window. "
            "Do NOT invent or extrapolate clock times.\n"
            "4) Key Data Features: extract 1–2 forecasting-critical characteristics supported by the data, such as "
            "wind speed persistence vs rapid ramps, hub-height speed variability, "
            "vertical shear between 30 m / 50 m / hub height (e.g., hub > 50 m > 30 m consistently), "
            "power saturation near rated output (power flat despite higher wind), "
            "or anomalies (e.g., power drops while wind remains high). Avoid vague descriptors.\n"
            "5) Forecasting Guidance (ACTIONABLE REQUIRED): map the inferred wind-regime timing and observed features onto the next 24 hours via relative extension. "
            "Continue the wind speed dynamics (persistence or rampiness) and apply a consistent wind-to-power relationship: "
            "if historical low-wind periods produce near-zero/very low power, enforce similarly low output in corresponding future low-wind periods; "
            "if power saturates near rated during high winds, maintain capped output under similar high-wind behavior. "
            "Make a clear assumption; avoid generic phrasing.\n\n"
        )

            
        # self.system = (
        #     "You are a photovoltaic data feature extractor. Your task is to convert a JSON summary of ONE "
        #     "24-hour historical window (sampled every 15 minutes) into a concise, objective English prompt "
        #     "that conditions a downstream PV power forecasting model.\n\n"
        #     "Output MUST be strict JSON only: {\"prompt_text\":\"...\"}. Do not include any content outside the JSON. "
        #     "Inside the JSON string, use \\n exactly once per line break.\n\n"
        #     "Write a focused 80–100 word prompt based ONLY on observed behavior in the input data. "
        #     "The prompt_text MUST include the following elements:\n"
        #     "1) Task: explicitly state that the goal is to forecast PV power for the next 24 hours.\n"
        #     "2) Time Window: report the exact start–end timestamps of the historical data window.\n"
        #     "3) Day/Night Timing (NUMERIC REQUIRED when identifiable): infer daytime and nighttime periods "
        #     "from sustained near-zero versus active generation/irradiance. "
        #     "When clearly identifiable, report explicit time ranges in HH:MM–HH:MM using ONLY times observed "
        #     "in the historical window. Do NOT invent or extrapolate clock times.\n"
        #     "4) Key Data Features: extract 1–2 forecasting-critical characteristics supported by the data, "
        #     "such as daytime smoothness vs variability, irradiance–power synchrony, persistence, or anomalies "
        #     "(e.g., abrupt midday drops). Avoid vague descriptors.\n"
        #     "5) Forecasting Guidance (ACTIONABLE REQUIRED): map the inferred day–night timing and observed "
        #     "features onto the next 24 hours via relative extension. "
        #     "Enforce near-zero output during the corresponding nighttime periods and continue the daytime "
        #     "generation profile with the same smoothness or variability observed historically. "
        #     "Make a clear assumption; avoid generic phrasing.\n\n"
        # )






    def _validate_obj(self, obj: Dict[str, Any]) -> str:
        """
        严格验收标准：
          1) 只能有一个 key：prompt_text
          2) prompt_text 必须是非空字符串
          3) prompt_text 必须包含 5 个前缀标记（防止只返回第一行/拆字段）
        """
        if not isinstance(obj, dict):
            raise RuntimeError(f"Parsed JSON is not an object: {type(obj)}")

        if set(obj.keys()) != {"prompt_text"}:
            raise RuntimeError(f"Unexpected keys: {list(obj.keys())} (must be only 'prompt_text')")

        prompt_text = obj.get("prompt_text", None)
        # if not isinstance(prompt_text, str) or not prompt_text.strip():
        #     raise RuntimeError("'prompt_text' missing or not a non-empty string")

        # required_markers: List[str] = [
        #     "PV power forecasting task.",
        #     "Window:",
        #     "Night/Day:",
        #     "Pattern:",
        #     "Goal:",
        # ]
        # missing = [m for m in required_markers if m not in prompt_text]
        # if missing:
        #     raise RuntimeError(f"prompt_text missing required markers: {missing}")

        return prompt_text

    def generate_prompt_text(self, window_summary: Dict[str, Any]) -> str:
        """
        不达标就重试；达标立即返回；超过 max_retries 抛异常
        """
        user_msg = json.dumps(window_summary, ensure_ascii=False)

        last_err: Optional[Exception] = None
        last_raw: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            kwargs = dict(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.95,
                presence_penalty=0.5,
                stream=False,
                extra_body={
                    "top_k": 40, 
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            if self.use_response_format:
                kwargs["response_format"] = {"type": "json_object"}

            try:
                resp = self.client.chat.completions.create(**kwargs)
                raw = resp.choices[0].message.content
                if raw is None:
                    raise RuntimeError("LLM returned empty content")
                last_raw = raw

                obj = json.loads(raw)          # ✅ 严格 JSON（不抽取/不修补）
                prompt_text = self._validate_obj(obj)  # ✅ 严格验收
                return prompt_text

            except Exception as e:
                last_err = e
                if self.retry_sleep > 0:
                    time.sleep(self.retry_sleep)
                continue

        raise RuntimeError(
            f"LLM output failed validation after {self.max_retries} retries.\n"
            f"Last error: {last_err}\n"
            f"Last raw output:\n{last_raw}"
        )
