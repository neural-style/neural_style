# coding:utf-8
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import time
from datetime import timedelta


# 设置允许的文件格式
ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'PNG', 'bmp'}

flag = [0]
flag_diy = [0]
flag_generate = []


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


@app.route('/', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        flag.append(1)

        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static\\images\\input', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)

        # 使用Opencv转换一下图片格式和名称
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static\\images\\input', 'input.jpg'), img)

        os.remove(upload_path)

        return render_template('upload.html', flag=flag[-1])
    return render_template('upload.html', flag=flag[-1])


@app.route('/candy', methods=['POST', 'GET'])
def candy():
    flag_generate.append('candy')
    if request.method == 'POST':
        flag.append(1)

        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static\\images\\input', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)

        # 使用Opencv转换一下图片格式和名称
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static\\images\\input', 'input.jpg'), img)

        os.remove(upload_path)

        return render_template('candy.html', flag=flag[-1])
    return render_template('candy.html', flag=flag[-1])


@app.route('/mosaic', methods=['POST', 'GET'])
def mosaic():
    flag_generate.append('mosaic')
    if request.method == 'POST':
        flag.append(1)

        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static\\images\\input', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)

        # 使用Opencv转换一下图片格式和名称
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static\\images\\input', 'input.jpg'), img)

        os.remove(upload_path)

        return render_template('mosaic.html', flag=flag[-1])
    return render_template('mosaic.html', flag=flag[-1])


@app.route('/rain_princess', methods=['POST', 'GET'])
def rain_princess():
    flag_generate.append('rain_princess')
    if request.method == 'POST':
        flag.append(1)

        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static\\images\\input', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)

        # 使用Opencv转换一下图片格式和名称
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static\\images\\input', 'input.jpg'), img)

        os.remove(upload_path)

        return render_template('rain_princess.html', flag=flag[-1])
    return render_template('rain_princess.html', flag=flag[-1])


@app.route('/udnie', methods=['POST', 'GET'])
def udnie():
    flag_generate.append('udnie')
    if request.method == 'POST':
        flag.append(1)

        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static\\images\\input', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)

        # 使用Opencv转换一下图片格式和名称
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static\\images\\input', 'input.jpg'), img)

        os.remove(upload_path)

        return render_template('udnie.html', flag=flag[-1])
    return render_template('udnie.html', flag=flag[-1])


@app.route('/diy', methods=['POST', 'GET'])
def diy():
    flag_generate.append('diy')
    if request.method == 'POST':
        flag_diy.append(1)

        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static\\images\\style', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)

        # 使用Opencv转换一下图片格式和名称
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static\\images\\style', 'diy.jpg'), img)
        os.remove(upload_path)

        return redirect(url_for('diy_ok'))
    return render_template('diy.html', flag=flag[-1], flag_diy=flag_diy[-1])


@app.route('/diy_ok', methods=['POST', 'GET'])
def diy_ok():
    if request.method == 'POST':
        flag.append(1)
        flag_generate.append('diy')

        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        user_input = request.form.get("name")

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path = os.path.join(basepath, 'static\\images\\input', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        f.save(upload_path)

        # 使用Opencv转换一下图片格式和名称
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static\\images\\input', 'input.jpg'), img)

        os.remove(upload_path)

        return render_template('diy.html', flag=flag[-1], flag_diy=flag_diy[-1])
    return render_template('diy.html', flag=flag[-1], flag_diy=flag_diy[-1])


@app.route('/generate', methods=['POST', 'GET'])
def generate():
    print(flag_generate)
    style = flag_generate[-1]

    return render_template('generate.html', style=style)


if __name__ == '__main__':
    # app.debug = True
    app.run(host='0.0.0.0', port=8987, debug=True)
