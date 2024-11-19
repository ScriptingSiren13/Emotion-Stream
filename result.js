const urlParams = new URLSearchParams(window.location.search);
const accessToken = urlParams.get('access_token');
const emotion = urlParams.get('emotion') || 'happy';

document.getElementById('detected-emotion').textContent = emotion;

if (!accessToken) {
    document.getElementById('status').textContent = 'Error: No access token found. Please authorize first.';
    throw new Error('No access token found');
}

let player;
let device_id;

function initializePlayer() {
    player = new Spotify.Player({
        name: 'Web Playback SDK Player',
        getOAuthToken: callback => { callback(accessToken); }
    });

    player.addListener('initialization_error', ({ message }) => { console.error(message); });
    player.addListener('authentication_error', ({ message }) => { console.error(message); });
    player.addListener('account_error', ({ message }) => { console.error(message); });
    player.addListener('playback_error', ({ message }) => { console.error(message); });

    player.addListener('ready', ({ device_id: d_id }) => {
        device_id = d_id;
        document.getElementById('status').textContent = 'Player is ready!';
        

        // Automatically play playlist based on emotion
        const playlistUri = getPlaylistUriBasedOnEmotion(emotion);
        startPlayback(device_id, playlistUri);

        // Enable control buttons
        document.getElementById('play-btn').disabled = false;
        document.getElementById('pause-btn').disabled = false;
        document.getElementById('next-btn').disabled = false;
        document.getElementById('prev-btn').disabled = false;
    });

    player.connect();
}

function getPlaylistUriBasedOnEmotion(emotion) {
    const emotionPlaylists = {
        'happy': 'spotify:playlist:4PGxMjK8uFc8N9suEWvfDZ',
        'sad': 'spotify:playlist:7rxJ5ST7w0dKYe0J7MGUuD',
        'angry': 'spotify:playlist:6yRvkkvLeTTO4BGdiPCjbo',
        'disgust': 'spotify:playlist:1AW5OqKLxBtMc177Yn28QA',
        'neutral': 'spotify:playlist:64QQ1iHIz6GIgS44YxWy6A',
        'surprise': 'spotify:playlist:1G7PCOSXsha4n6mmP1XKZJ',
        'fear': 'spotify:playlist:3BPeUJe6fNt7eiG9MLJ301'
    };
    return emotionPlaylists[emotion] || emotionPlaylists['happy'];
}

function startPlayback(device_id, playlistUri) {
    fetch(`https://api.spotify.com/v1/me/player/play?device_id=${device_id}`, {
        method: 'PUT',
        body: JSON.stringify({ context_uri: playlistUri }),
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${accessToken}`
        }
    }).then(response => {
        if (response.ok) {
            console.log('Playback started successfully!');
            document.getElementById('status').textContent = 'Playing...';
            updateNowPlaying();
        } else {
            document.getElementById('status').textContent = 'Error starting playback.';
            console.error('Error:', response);
        }
    }).catch(error => {
        document.getElementById('status').textContent = 'Error during playback.';
        console.error('Error during playback:', error);
    });
}

// Enhanced control buttons
document.getElementById('play-btn').addEventListener('click', () => {
    player.resume().then(() => {
        document.getElementById('play-btn').disabled = true;
        document.getElementById('pause-btn').disabled = false;
        updateNowPlaying();
    }).catch(error => console.error('Error playing:', error));
});

document.getElementById('pause-btn').addEventListener('click', () => {
    player.pause().then(() => {
        document.getElementById('play-btn').disabled = false;
        document.getElementById('pause-btn').disabled = true;
    }).catch(error => console.error('Error pausing:', error));
});

document.getElementById('prev-btn').addEventListener('click', () => {
    player.previousTrack().then(() => updateNowPlaying()).catch(error => console.error('Error going to previous track:', error));
});

document.getElementById('next-btn').addEventListener('click', () => {
    player.nextTrack().then(() => updateNowPlaying()).catch(error => console.error('Error going to next track:', error));
});

function updateNowPlaying() {
    player.getCurrentState().then(state => {
        if (!state || !state.track_window.current_track) {
            document.getElementById('now-playing').textContent = 'Not playing anything.';
            return;
        }

        const track = state.track_window.current_track;
        const trackName = track.name;
        const artistNames = track.artists.map(artist => artist.name).join(', ');
        document.getElementById('now-playing').textContent = `Now Playing: ${trackName} by ${artistNames}`;
        
        // Update every 5 seconds
        setTimeout(() => updateNowPlaying(), 5000);
    });
}

// Search for songs and podcasts
document.getElementById('search-btn').addEventListener('click', () => {
const query = document.getElementById('search-input').value.trim();

if (query === '') {
    document.getElementById('search-results').innerHTML = 'Please enter a search term.';
    return;
}

searchSongsAndPodcasts(query);
});

function searchSongsAndPodcasts(query) {
fetch(`https://api.spotify.com/v1/search?q=${encodeURIComponent(query)}&type=track,show&limit=5`, {
    headers: {
        'Authorization': `Bearer ${accessToken}`
    }
})
.then(response => {
    if (!response.ok) {
        throw new Error('Failed to fetch search results');
    }
    return response.json();
})
.then(data => {
    const resultsContainer = document.getElementById('search-results');
    resultsContainer.innerHTML = ''; // Clear previous results

    // If no results found
    if (data.tracks.items.length === 0 && data.shows.items.length === 0) {
        resultsContainer.innerHTML = 'No results found.';
        return;
    }

    // Handle track results
    if (data.tracks.items.length > 0) {
        const trackHeader = document.createElement('h3');
        trackHeader.textContent = 'Tracks:';
        resultsContainer.appendChild(trackHeader);

        data.tracks.items.forEach(track => {
            const trackElement = document.createElement('div');
            trackElement.className = 'search-result';
            trackElement.textContent = `${track.name} by ${track.artists.map(artist => artist.name).join(', ')}`;
            trackElement.onclick = () => playTrack(track.uri); // Play the track on click
            resultsContainer.appendChild(trackElement);
        });
    }

    // Handle podcast (show) results
    if (data.shows.items.length > 0) {
        const showHeader = document.createElement('h3');
        showHeader.textContent = 'Podcasts:';
        resultsContainer.appendChild(showHeader);

        data.shows.items.forEach(show => {
            const showElement = document.createElement('div');
            showElement.className = 'search-result';
            showElement.textContent = `Podcast: ${show.name} by ${show.publisher}`;
            showElement.onclick = () => playPodcastEpisode(show.id); // Play latest episode on click
            resultsContainer.appendChild(showElement);
        });
    }
})
.catch(error => {
    console.error('Error fetching search results:', error);
    document.getElementById('search-results').innerHTML = 'Error fetching search results.';
});
}

function playTrack(uri) {
fetch(`https://api.spotify.com/v1/me/player/play`, {
    method: 'PUT',
    body: JSON.stringify({ uris: [uri] }),
    headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${accessToken}`
    }
}).then(response => {
    if (response.ok) {
        updateNowPlaying(); // Update the now playing info
    } else {
        console.error('Error playing track:', response);
        document.getElementById('status').textContent = 'Error playing track.';
    }
}).catch(error => {
    console.error('Error playing track:', error);
});
}

function playPodcastEpisode(showId) {
// Fetch the latest episode of the podcast
fetch(`https://api.spotify.com/v1/shows/${showId}/episodes?limit=1`, {
    headers: {
        'Authorization': `Bearer ${accessToken}`
    }
})
.then(response => {
    if (!response.ok) {
        throw new Error('Failed to fetch podcast episodes');
    }
    return response.json();
})
.then(data => {
    const episodeUri = data.items[0].uri;
    fetch(`https://api.spotify.com/v1/me/player/play`, {
        method: 'PUT',
        body: JSON.stringify({ uris: [episodeUri] }),
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${accessToken}`
        }
    }).then(response => {
        if (response.ok) {
            updateNowPlaying(); // Update the now playing info
        } else {
            console.error('Error playing podcast episode:', response);
            document.getElementById('status').textContent = 'Error playing podcast episode.';
        }
    }).catch(error => {
        console.error('Error playing podcast episode:', error);
    });
})
.catch(error => {
    console.error('Error fetching podcast episodes:', error);
    document.getElementById('search-results').innerHTML = 'Error fetching podcast episodes.';
});
}

function updateNowPlaying() {
player.getCurrentState().then(state => {
    if (!state || !state.track_window.current_track) {
        document.getElementById('now-playing').textContent = 'Not playing anything.';
        return;
    }

    const track = state.track_window.current_track;
    const trackName = track.name;
    const artistNames = track.artists.map(artist => artist.name).join(', ');
    document.getElementById('now-playing').textContent = `Now Playing: ${trackName} by ${artistNames}`;
    
    // Update every 5 seconds
    setTimeout(() => updateNowPlaying(), 5000);
});
}

document.getElementById('back-home-btn').addEventListener('click', () => {
    window.location.href = '/';
});

function logoutFromSpotify() {
    var logoutWindow = window.open('https://accounts.spotify.com/en/logout', '_blank');

    setTimeout(function() {
        if (logoutWindow) {
            logoutWindow.close();
        }
        window.location.href = '/logout';
    }, 2000);
}

window.onload = initializePlayer; // Initialize the player when the page loads