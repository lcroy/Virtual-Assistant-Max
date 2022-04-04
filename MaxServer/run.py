from flask import Flask, request, jsonify, render_template, send_file
from services.language_service.intent_recognition.max_predictor import *
import json
import os


app = Flask(__name__)
app.add_url_rule('/photos/<path:filename>', ...)
tasks = []

#update the
@app.route('/get_conv/', methods=['GET'])
def get_conv():
    filename = os.path.join(app.static_folder, 'data', 'conv.json')
    with open(filename) as json_file:
        data = json.load(json_file)

    return jsonify(data)


# calling the language service + robot service
@app.route('/get_service/', methods=['GET'])
def get_service():
    # bad requests
    if not request.args or 'message' not in request.args:
        return jsonify(
            service_name = 'none',
            intent='none',
            slot='none',
            required_slot = [],
            result= 'bad_request'
        )
    else:
        user_utterance = request.args.get('message')
        requested_service = request.args.get('requested_service')
        client_slot_result = json.loads(request.args.get('client_slot_result'))
        if len(user_utterance) > 0:

            # Predict the requested intent
            pred_result, pred_service, language_service_result_flag, updated_client_slot_result, required_slot_list = pred_intent_slot(user_utterance, client_slot_result, requested_service)

            # Search the service (robot service or other)
            # Max don't understand
            if language_service_result_flag == 0:
                return jsonify(
                    service_name=requested_service,
                    intent='none',
                    slot='none',
                    required_slot=[],
                    result='do_not_understand'
                )
            # everything is fine
            else:

                if pred_service == 'main':
                    return jsonify(
                        service_name=pred_service,
                        intent=pred_result['intent'],
                        slot=updated_client_slot_result,
                        required_slot=required_slot_list,
                        result='good_result'
                    )
                else:
                    return jsonify(
                        service_name=pred_service,
                        intent=pred_result['intent'],
                        slot=updated_client_slot_result,
                        required_slot=required_slot_list,
                        result='good_result'
                    )
        # user's utterance is empty
        else:
            return jsonify(
                service_name='none',
                intent='none',
                slot=client_slot_result,
                required_slot=[],
                result='did_not_catch'
            )

# calling the index
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# download the
@app.route('/download', methods=['GET'])
def downloadFile ():
    path = os.path.join(app.static_folder, 'data', 'service_list.json')
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
