# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

# Third Party
import gradio as gr
import grpc


class MaskGeneration:
    def __init__(self, request, predict) -> None:
        self.request = request
        self.predict = predict

    def fn(self, model, text_in):
        # False-y string doesn't work as required request param so '' --> ''
        response = self.predict(
            self.request(text_in=text_in), metadata=[("mm-model-id", model)]
        )
        results = list(response.objects)
        return results[0].score, results[0].token_str, results[0].sequence, results[1].score, results[1].token_str, results[1].sequence


    @classmethod
    def optional_tab(cls, models, request, predict):
        if not models:
            return False

        tab = cls.__name__  # tab name
        try:
            this = cls(request, predict)
            with gr.Tab(tab):
                model_choice = gr.Dropdown(
                    label="Model ID", choices=models, value=models[0]
                )
                inputs = gr.Textbox(
                    label="Input Text (hit enter to send; must include <mask>)", placeholder=f"Enter input text for {tab}"
                )
                with gr.Row():
                    with gr.Column():
                        output_score1 = gr.Textbox(
                            label="Score", placeholder=f""
                        )
                    with gr.Column():
                        output_token_str1 = gr.Textbox(
                            label="Token", placeholder=f""
                        )
                    with gr.Column():
                        output_sequence1 = gr.Textbox(
                            label="Sequence", placeholder=f""
                        )
                with gr.Row():
                    with gr.Column():
                        output_score2 = gr.Textbox(
                            label="Score", placeholder=f""
                        )
                    with gr.Column():
                        output_token_str2 = gr.Textbox(
                            label="Token", placeholder=f""
                        )
                    with gr.Column():
                        output_sequence2 = gr.Textbox(
                            label="Sequence", placeholder=f""
                        )
                inputs.submit(this.fn, [model_choice, inputs], [output_score1, output_token_str1, output_sequence1, output_score2, output_token_str2, output_sequence2], api_name=tab)
                model_choice.change(
                    this.fn, [model_choice, inputs], [output_score1, output_token_str1, output_sequence1, output_score2, output_token_str2, output_sequence2], api_name=tab
                )
                print(f"✅️  {tab} tab is enabled!")
                return True
        except grpc.RpcError as rpc_error:
            print(f"⚠️  Disabling {tab} tab due to:  {rpc_error.details()}")
            return False
