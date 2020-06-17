#from keras import applications
import flask
from flask import Flask, request, render_template
import pickle
import numpy as np



# Use pickle to load in the pre-trained model.
with open(f'model/NN_model_rent_pred.pkl', 'rb') as f:
    model = pickle.load(f)
app = flask.Flask(__name__, template_folder='templates')
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        data_raw=[0] * 40
        data_raw=[(x) for x in request.form.values()]
        x_test=[0] * 40
        x_test[0:10]=data_raw[0:10]


        if any("condition_0" in s for s in data_raw):
            x_test[15]=1
        if any("condition_Heas korras" in s for s in data_raw):
            x_test[16]=1
        if any("condition_Keskmine" in s for s in data_raw):
            x_test[17]=1
        if any("condition_Renoveeritud" in s for s in data_raw):
            x_test[18]=1
        if any("condition_San. remont tehtud" in s for s in data_raw):
            x_test[19]=1           
        if any("condition_Vajab renoveerimist" in s for s in data_raw):
            x_test[20]=1
        if any("condition_Vajab san. remonti" in s for s in data_raw):
            x_test[21]=1
        if any("condition_Valmis" in s for s in data_raw):
            x_test[22]=1
        if any("condition_Vajab renoveerimist" in s for s in data_raw):
            x_test[23]=1




        if any("en_mark_0" in s for s in data_raw):
            x_test[24]=1
        if any("en_mark_A" in s for s in data_raw):
            x_test[25]=1
        if any("en_mark_B" in s for s in data_raw):
            x_test[26]=1
        if any("en_mark_C" in s for s in data_raw):
            x_test[27]=1           
        if any("en_mark_D" in s for s in data_raw):
            x_test[28]=1
        if any("en_mark_E" in s for s in data_raw):
            x_test[29]=1
        if any("en_mark_F" in s for s in data_raw):
            x_test[30]=1
        if any("en_mark_G" in s for s in data_raw):
            x_test[31]=1
        if any("en_mark_H" in s for s in data_raw):
            x_test[32]=1
        if any("en_mark_P" in s for s in data_raw):
            x_test[33]=1


        if any("parking_0" in s for s in data_raw):
            x_test[34]=1
        if any("parking_maja" in s for s in data_raw):
            x_test[35]=1
        if any("parking_tasuline" in s for s in data_raw):
            x_test[36]=1
        if any("parking_tasuta" in s for s in data_raw):
            x_test[37]=1

        x_test_fin = np.array(x_test,dtype="float64")
        x_test_fin_re=x_test_fin.reshape(-1,1)
        x_test_fin_re_t=x_test_fin_re.T

        x_pred=model.predict(x_test_fin_re_t)
        
        return flask.render_template('main.html', result=x_pred, original_input=x_test_fin_re_t)
if __name__ == '__main__':
    app.run()
