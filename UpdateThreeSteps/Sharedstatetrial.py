#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 10:14:50 2025

@author: ctlewis
"""

import panel as pn
import param
import datetime
from threading import Thread
import time

pn.extension(sizing_mode="stretch_width")
class StreamClass(param.Parameterized):
    value = param.Integer()
class MessageQueue(param.Parameterized):
    value = param.List()

    def append(self, asof, user, message):
        if message:
            self.value = [*self.value, (asof, user, message)]

ACCENT_COLOR = "#0072B5"
DEFAULT_PARAMS = {
    "site": "Panel Multi Page App",
    "accent_base_color": ACCENT_COLOR,
    "header_background": ACCENT_COLOR,
}

def fastlisttemplate(title, *objects):
    """Returns a Panel-AI version of the FastListTemplate

    Returns:
        [FastListTemplate]: A FastListTemplate
    """
    return pn.template.FastListTemplate(**DEFAULT_PARAMS, title=title, main=[pn.Column(*objects)])


def get_shared_state():
    if not "stream" in pn.state.cache:
        state=pn.state.cache["stream"]=StreamClass()
        pn.state.cache["messages"]=MessageQueue()

        def update_state():
            while True:
                if state.value==100:
                    state.value=0
                else:
                    state.value+=1
                time.sleep(1)

        Thread(target=update_state).start()

    return pn.state.cache["stream"], pn.state.cache["messages"]

def show_messages(messages):
    result = ""
    for message in messages:
        result = f"- {message[0]} | {message[1]}: {message[2]}\n" + result
    if not result:
        result = "No Messages yet!"
    return result

def page1():
    _, messages = get_shared_state()

    user_input = pn.widgets.TextInput(value="Guest", name="User")
    message_input = pn.widgets.TextInput(value="Hello", name="Message")
    add_message_button = pn.widgets.Button(name="Add")

    def add_message(event):
        messages.append(datetime.datetime.utcnow(), user_input.value, message_input.value)
    add_message_button.on_click(add_message)

    return fastlisttemplate("Add Message", user_input, message_input, add_message_button)

def page2():
    _, messages = get_shared_state()

    ishow_messages = pn.bind(show_messages, messages=messages.param.value)
    return fastlisttemplate("Show Messages",pn.panel(ishow_messages, height=600),)

def page3():
    stream, _ = get_shared_state()

    return fastlisttemplate("Show Streaming Value",stream.param.value,)

ROUTES = {
    "1": page1, "2": page2, "3": page3
}
pn.serve(ROUTES, port=5006, autoreload=True)

