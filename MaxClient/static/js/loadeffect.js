function myFunction(){
            const wave = new Wave()
            let mediaStream = null;
            let visuals = ["flower"]
            let visualSelector = document.getElementById("visuals")
            visuals.forEach((visual) => {
                let option = document.createElement("option")
                option.value = visual
                option.innerText = visual
                option.selected = true
                visualSelector.appendChild(option)
            })
            if (mediaStream == null) {
                mainCanvas.classList.add("show")
                mainCanvas.height = window.innerHeight * .8
                mainCanvas.width = window.innerWidth * .8

                navigator.mediaDevices.getUserMedia({
                    audio: true
                })
                .then(function (stream) {
                    mediaStream = stream
                    wave.fromStream(mediaStream, "mainCanvas", { type: visualSelector.options[visualSelector.selectedIndex].value }, false)
                })
                .catch(function (err) {
                    console.log(err.message)
                })
            } else {
                wave.fromStream(mediaStream, "mainCanvas", { type: visualSelector.options[visualSelector.selectedIndex].value }, false)
            }

        };

var max_text_temp = ''
var user_text_temp = ''
var frank_internet_text_temp = ''
var frank_sys_text_temp = ''
var frank_task_text_temp = ''
var mir_internet_text_temp = ''
var mir_sys_text_temp = ''
var mir_task_text_temp = ''
var swarm_internet_text_temp = ''
var swarm_sys_text_temp = ''
var swarm_task_text_temp = ''

var typeWriter_max = {
    msg: function(msg){
     return msg;
    },
    len: function(){
     return this.msg.length;
    },
    seq: 0,
    speed: 65,//打字时间(ms)
    type: function(){
     var _this = this;
     document.getElementById("Max").innerHTML = "Max: " + _this.msg.substring(0, _this.seq);
     if (_this.seq == _this.len()) {
      _this.seq = 0;
       clearTimeout(t);
     }
     else {
      _this.seq++;
      var t = setTimeout(function(){_this.type()}, this.speed);
     }
    }
   };

var typeWriter_user = {
    msg: function(msg){
     return msg;
    },
    len: function(){
     return this.msg.length;
    },
    seq: 0,
    speed: 10,//打字时间(ms)
    type: function(){
     var _this = this;
     document.getElementById("User").innerHTML = "User: " + _this.msg.substring(0, _this.seq);
     if (_this.seq == _this.len()) {
      _this.seq = 0;
       clearTimeout(t);
     }
     else {
      _this.seq++;
      var t = setTimeout(function(){_this.type()}, this.speed);
     }
    }
   };

var typeWriter_franka_task = {
    msg: function(msg){
     return msg;
    },
    len: function(){
     return this.msg.length;
    },
    seq: 0,
    speed: 10,//打字时间(ms)
    type: function(){
     var _this = this;
     document.getElementById("Franka_task").innerHTML = "<span style='color: #29b830;'>Task: </span>" + _this.msg.substring(0, _this.seq);
     if (_this.seq == _this.len()) {
      _this.seq = 0;
       clearTimeout(t);
     }
     else {
      _this.seq++;
      var t = setTimeout(function(){_this.type()}, this.speed);
     }
    }
   };

var typeWriter_franka_internet = {
    msg: function(msg){
     return msg;
    },
    len: function(){
     return this.msg.length;
    },
    seq: 0,
    speed: 10,//打字时间(ms)
    type: function(){
     var _this = this;
     document.getElementById("Franka_internet").innerHTML = "<span style='color: #29b830;'>Internet Connection: </span>" + _this.msg.substring(0, _this.seq);
     if (_this.seq == _this.len()) {
      _this.seq = 0;
       clearTimeout(t);
     }
     else {
      _this.seq++;
      var t = setTimeout(function(){_this.type()}, this.speed);
     }
    }
   };

var typeWriter_franka_sys = {
    msg: function(msg){
     return msg;
    },
    len: function(){
     return this.msg.length;
    },
    seq: 0,
    speed: 10,//打字时间(ms)
    type: function(){
     var _this = this;
     document.getElementById("Franka_sys").innerHTML = "<span style='color: #29b830;'>System Status: </span>" + _this.msg.substring(0, _this.seq);
     if (_this.seq == _this.len()) {
      _this.seq = 0;
       clearTimeout(t);
     }
     else {
      _this.seq++;
      var t = setTimeout(function(){_this.type()}, this.speed);
     }
    }
   };

var typeWriter_mir_task = {
    msg: function(msg){
     return msg;
    },
    len: function(){
     return this.msg.length;
    },
    seq: 0,
    speed: 10,//打字时间(ms)
    type: function(){
     var _this = this;
     document.getElementById("mir_task").innerHTML = "<span style='color: #29b830;'>Task: </span>" + _this.msg.substring(0, _this.seq);
     if (_this.seq == _this.len()) {
      _this.seq = 0;
       clearTimeout(t);
     }
     else {
      _this.seq++;
      var t = setTimeout(function(){_this.type()}, this.speed);
     }
    }
   };

var typeWriter_mir_internet = {
    msg: function(msg){
     return msg;
    },
    len: function(){
     return this.msg.length;
    },
    seq: 0,
    speed: 10,//打字时间(ms)
    type: function(){
     var _this = this;
     document.getElementById("mir_internet").innerHTML = "<span style='color: #29b830;'>Internet Connection: </span>" + _this.msg.substring(0, _this.seq);
     if (_this.seq == _this.len()) {
      _this.seq = 0;
       clearTimeout(t);
     }
     else {
      _this.seq++;
      var t = setTimeout(function(){_this.type()}, this.speed);
     }
    }
   };

var typeWriter_mir_sys = {
    msg: function(msg){
     return msg;
    },
    len: function(){
     return this.msg.length;
    },
    seq: 0,
    speed: 10,//打字时间(ms)
    type: function(){
     var _this = this;
     document.getElementById("mir_sys").innerHTML = "<span style='color: #29b830;'>System Status: </span>" + _this.msg.substring(0, _this.seq);
     if (_this.seq == _this.len()) {
      _this.seq = 0;
       clearTimeout(t);
     }
     else {
      _this.seq++;
      var t = setTimeout(function(){_this.type()}, this.speed);
     }
    }
   };

var typeWriter_swarm_task = {
    msg: function(msg){
     return msg;
    },
    len: function(){
     return this.msg.length;
    },
    seq: 0,
    speed: 10,//打字时间(ms)
    type: function(){
     var _this = this;
     document.getElementById("swarm_task").innerHTML = "<span style='color: #29b830;'>Task: </span>" + _this.msg.substring(0, _this.seq);
     if (_this.seq == _this.len()) {
      _this.seq = 0;
       clearTimeout(t);
     }
     else {
      _this.seq++;
      var t = setTimeout(function(){_this.type()}, this.speed);
     }
    }
   };

var typeWriter_swarm_internet = {
    msg: function(msg){
     return msg;
    },
    len: function(){
     return this.msg.length;
    },
    seq: 0,
    speed: 10,//打字时间(ms)
    type: function(){
     var _this = this;
     document.getElementById("swarm_internet").innerHTML = "<span style='color: #29b830;'>Internet Connection: </span>" + _this.msg.substring(0, _this.seq);
     if (_this.seq == _this.len()) {
      _this.seq = 0;
       clearTimeout(t);
     }
     else {
      _this.seq++;
      var t = setTimeout(function(){_this.type()}, this.speed);
     }
    }
   };

var typeWriter_swarm_sys = {
    msg: function(msg){
     return msg;
    },
    len: function(){
     return this.msg.length;
    },
    seq: 0,
    speed: 10,//打字时间(ms)
    type: function(){
     var _this = this;
     document.getElementById("swarm_sys").innerHTML = "<span style='color: #29b830;'>System Status: </span>" + _this.msg.substring(0, _this.seq);
     if (_this.seq == _this.len()) {
      _this.seq = 0;
       clearTimeout(t);
     }
     else {
      _this.seq++;
      var t = setTimeout(function(){_this.type()}, this.speed);
     }
    }
   };


function get_conv() {
            $.ajax({
                url:'/get_conv/',
                type:"GET",
                dataType:'json',
                success:function (data) {
                    $.each(data,function(k,v) {
                        if (v.service == 'home'){
                            document.getElementById("welcome").hidden = false;
                            document.getElementById("mir_S").hidden = true;
                            document.getElementById("franka_S").hidden = true;
                            document.getElementById("swarm_S").hidden = true;
                            document.getElementById("robot_status").style.visibility = 'hidden';
                            document.getElementById("Franka_internet").hidden = true;
                            document.getElementById("Franka_sys").hidden = true;
                            document.getElementById("Franka_task").hidden = true;
                            document.getElementById("mir_internet").hidden = true;
                            document.getElementById("mir_sys").hidden = true;
                            document.getElementById("mir_task").hidden = true;
                            document.getElementById("swarm_internet").hidden = true;
                            document.getElementById("swarm_sys").hidden = true;
                            document.getElementById("swarm_task").hidden = true;
                        }else if (v.service == 'mir'){
                            document.getElementById("welcome").hidden = true;
                            document.getElementById("mir_S").hidden = false;
                            document.getElementById("robot_status").style.visibility = 'visible';
                            document.getElementById("franka_S").hidden = true;
                            document.getElementById("swarm_S").hidden = true;
                            document.getElementById("Franka_internet").hidden = true;
                            document.getElementById("Franka_sys").hidden = true;
                            document.getElementById("Franka_task").hidden = true;
                            document.getElementById("mir_internet").hidden = false;
                            document.getElementById("mir_sys").hidden = false;
                            document.getElementById("mir_task").hidden = false;
                            document.getElementById("swarm_internet").hidden = true;
                            document.getElementById("swarm_sys").hidden = true;
                            document.getElementById("swarm_task").hidden = true;
                        }
                        else if (v.service == 'franka'){
                            document.getElementById("welcome").hidden = true;
                            document.getElementById("mir_S").hidden = true;
                            document.getElementById("swarm_S").hidden = true;
                            document.getElementById("robot_status").style.visibility = 'visible';
                            document.getElementById("franka_S").hidden = false;
                            document.getElementById("Franka_internet").hidden = false;
                            document.getElementById("Franka_sys").hidden = false;
                            document.getElementById("Franka_task").hidden = false;
                            document.getElementById("mir_internet").hidden = true;
                            document.getElementById("mir_sys").hidden = true;
                            document.getElementById("mir_task").hidden = true;
                            document.getElementById("swarm_internet").hidden = true;
                            document.getElementById("swarm_sys").hidden = true;
                            document.getElementById("swarm_task").hidden = true;
                        }
                        else if (v.service == 'swarm'){
                            document.getElementById("welcome").hidden = true;
                            document.getElementById("mir_S").hidden = true;
                            document.getElementById("robot_status").style.visibility = 'visible';
                            document.getElementById("franka_S").hidden = true;
                            document.getElementById("swarm_S").hidden = false;
                            document.getElementById("Franka_internet").hidden = true;
                            document.getElementById("Franka_sys").hidden = true;
                            document.getElementById("Franka_task").hidden = true;
                            document.getElementById("mir_internet").hidden = true;
                            document.getElementById("mir_sys").hidden = true;
                            document.getElementById("mir_task").hidden = true;
                            document.getElementById("swarm_internet").hidden = false;
                            document.getElementById("swarm_sys").hidden = false;
                            document.getElementById("swarm_task").hidden = false;
                        }
                        var max_text = v.Max
                        var user_text = v.User
                        var franka_internet = v.franka_internet
                        var frank_sys = v.franka_system
                        var franka_task = v.franka_task
                        var mir_internet = v.mir_internet
                        var mir_sys = v.mir_system
                        var mir_task = v.mir_task
                        var swarm_internet = v.swarm_internet
                        var swarm_sys = v.swarm_system
                        var swarm_task = v.swarm_task

                        if (max_text != max_text_temp) {
                            typeWriter_max.msg = max_text;
                            typeWriter_max.type();
                            max_text_temp = max_text;
                        }
                        if (user_text != user_text_temp) {
                            typeWriter_user.msg = user_text;
                            typeWriter_user.type();
                            user_text_temp = user_text;
                        }
                        if (v.service == 'franka'){

                            if(franka_internet != frank_internet_text_temp){
                                typeWriter_franka_internet.msg = franka_internet;
                                typeWriter_franka_internet.type();
                                frank_internet_text_temp = franka_internet;
                            }
                            if(frank_sys != frank_sys_text_temp){
                                typeWriter_franka_sys.msg = frank_sys;
                                typeWriter_franka_sys.type();
                                frank_sys_text_temp = frank_sys;
                            }
                            if(franka_task != frank_task_text_temp){
                                typeWriter_franka_task.msg = franka_task;
                                typeWriter_franka_task.type();
                                frank_task_text_temp = franka_task;
                            }
                        }

                        if (v.service == 'mir'){

                            if(mir_internet != mir_internet_text_temp){
                                typeWriter_mir_internet.msg = mir_internet;
                                typeWriter_mir_internet.type();
                                mir_internet_text_temp = mir_internet;
                            }
                            if(mir_sys != mir_sys_text_temp){
                                typeWriter_mir_sys.msg = mir_sys;
                                typeWriter_mir_sys.type();
                                mir_sys_text_temp = mir_sys;
                            }
                            if(mir_task != mir_task_text_temp){
                                typeWriter_mir_task.msg = mir_task;
                                typeWriter_mir_task.type();
                                mir_task_text_temp = mir_task;
                            }
                        }

                        if (v.service == 'swarm'){

                            if(swarm_internet != swarm_internet_text_temp){
                                typeWriter_swarm_internet.msg = swarm_internet;
                                typeWriter_swarm_internet.type();
                                swarm_internet_text_temp = swarm_internet;
                            }
                            if(swarm_sys != swarm_sys_text_temp){
                                typeWriter_swarm_sys.msg = swarm_sys;
                                typeWriter_swarm_sys.type();
                                swarm_sys_text_temp = swarm_sys;
                            }
                            if(swarm_task != swarm_task_text_temp){
                                typeWriter_swarm_task.msg = swarm_task;
                                typeWriter_swarm_task.type();
                                swarm_task_text_temp = swarm_task;
                            }
                        }
                    })
                }
            })
        };

setInterval(get_conv,1000);
//
// function showLocale(objD)
// {
//     var yy = objD.getYear();
//     if(yy<1900) yy = yy + 1900;
//     var MM = objD.getMonth()+1;
//     if(MM<10) MM = '0' + MM;
//     var dd = objD.getDate();
//     if(dd<10) dd = '0' + dd;
//     var hh = objD.getHours();
//     if(hh<10) hh = '0' + hh;
//     var mm = objD.getMinutes();
//     if(mm<10) mm = '0' + mm;
//     var ss = objD.getSeconds();
//     if(ss<10) ss = '0' + ss;
//     var ww = objD.getDay();
//     if (ww==0) ww="Sunday";
//     if (ww==1) ww="Monday";
//     if (ww==2) ww="Tuesday";
//     if (ww==3) ww="Wednesday";
//     if (ww==4) ww="Thursday";
//     if (ww==5) ww="Friday";
//     if (ww==6) ww="Saturday";
//     var time = hh + ":" + mm + ":" + ss;
//     var day = yy + "-" + MM + "-" + dd + " " + ww;
//     var str = [time, day];
//
//     return(str);
// }
//
// function tick()
// {
//     var today;
//     today = new Date();
//     str = showLocale(today);
//     document.getElementById("localtime").innerHTML = "Time: " + str[0];
//     document.getElementById("localday").innerHTML = "Date: " + str[1];
//     window.setTimeout("tick()", 1000);
// }
//
// tick();