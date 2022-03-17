const cb=document.getElementById("chatbot");
const tb=document.getElementById("text-area");
const messages=document.getElementById("messages");
const pu=document.getElementById("picupload");
const body=document.getElementsByTagName('body');
const days=['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
function Scroll()
{
  messages.scrollTo(0,messages.scrollHeight);
}
// used to creating sent message in UI
function message_send()
{
    let message=tb.value;
   
    if(message.length>0)
    {let msg_container=document.createElement("div");
    msg_container.classList.add("msg-container");
    msg_container.innerHTML="<div class='sent-messages'>"+message+"</div>";
    let time_span=document.createElement("div");
    time_span.classList.add("time-date");
    time_span.innerHTML=days[new Date().getDay()]+" "+new Date().getHours()+":"+new Date().getMinutes();
    
    messages.appendChild(msg_container);
    messages.appendChild(time_span);
    messages.scrollTo(0,messages.scrollHeight);
    tb.value="";

}



}
//event listener for detecting  enter press and sending message
document.addEventListener("keyup", function(event) {
    if (event.keyCode === 13) {
     
      message_send();
    }
  });
  function cloose()
  {
     cb.style.display="none";
  }
  function puclk()
  {
     pu.click();
     
  }
  //preview birth
  function preview_birth(urll)
  {
    let pr=document.createElement("div");
    pr.classList.add("sent-img-preview");
    pr.innerHTML=" <img class='img-for-preview' src='"+urll+"' alt=''><i class='material-icons close-preview'>close</i>";
    body.appendChild(pr);

  }



  //event listener for sending an image
  pu.addEventListener("change",function(e){
      let urll=URL.createObjectURL(e.target.files[0]);
      let append_img=document.createElement("div");
      let sent_image=document.createElement("img");
      sent_image.classList.add("sent-image");
      sent_image.src=urll;
      // preview_birth(urll);
      sent_image.alt="";
      append_img.classList.add("msg-container");
      append_img.appendChild(sent_image);
      messages.appendChild(append_img);
      let time_span=document.createElement("div");
    time_span.classList.add("time-date");
    time_span.innerHTML=days[new Date().getDay()]+" "+new Date().getHours()+":"+new Date().getMinutes();
    messages.appendChild(time_span);
      messages.scrollTo(0,messages.scrollHeight);

  });