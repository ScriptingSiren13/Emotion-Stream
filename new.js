let mediaRecorder;
let audioChunks = [];
let audioBlob;

document.getElementById("start-recording").addEventListener("click", async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.start();
    document.getElementById("start-recording").disabled = true;
    document.getElementById("stop-recording").disabled = false;

    mediaRecorder.addEventListener("dataavailable", event => {
        audioChunks.push(event.data);
    });

    mediaRecorder.addEventListener("stop", () => {
        audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        audioChunks = [];
        const audioFile = new File([audioBlob], "recording.wav", { type: "audio/wav" });

        const fileInput = document.getElementById("audioFile");
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(audioFile);
        fileInput.files = dataTransfer.files;

        document.getElementById("submitRecording").disabled = false;
    });
});

document.getElementById("stop-recording").addEventListener("click", () => {
    mediaRecorder.stop();
    document.getElementById("start-recording").disabled = false;
    document.getElementById("stop-recording").disabled = true;
});

// Show the record section if the user is authorized
window.onload = function() {
const recordSection = document.getElementById('record');
const isAuthorized = recordSection.dataset.authorized === 'true';  // Convert to boolean

console.log("Is Authorized:", isAuthorized);  // For debugging
if (isAuthorized) {
    // Show a popup message
    alert("You are now authorized!");

    // Delay before revealing the record section
    setTimeout(() => {
        recordSection.classList.remove('hidden');
        document.getElementById('authorization').classList.add('hidden');
        document.getElementById("logout-section").classList.remove("hidden"); // Hide authorization section
    }, 1000); // Delay of 2000 milliseconds (2 seconds)
} else {
    console.log("User is not authorized.");
}
};
//HIDDEN iFRAME

function logoutFromSpotify() {
// Open a new window to log out from Spotify
var logoutWindow = window.open('https://accounts.spotify.com/en/logout', '_blank');

// Optionally wait for a moment before redirecting to the main page
setTimeout(function() {
// Close the logout window after a short delay
if (logoutWindow) {
    logoutWindow.close();
}
// Redirect to the Flask logout route
window.location.href = '/logout';  // Your Flask logout route
}, 2000);  // Adjust the timeout as necessary
}
function recordfun(){
window.location.href = '/reco';
}