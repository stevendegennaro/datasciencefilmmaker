var map;
var markers = [];
var PAmarkers = [];

let api_key;
fetch('google_api_key.txt')
  .then(response => response.text())
  .then((data) => {
    api_key = data.trim();
    var scriptFileName = "https://maps.googleapis.com/maps/api/js?key=" + api_key + "&callback=initMap";
    var scriptElement = document.getElementById("load_google_maps_api");
    scriptElement.src = scriptFileName;
    document.head.appendChild(scriptElement);
  })
  .catch(error => {
    console.error("Error fetching the API key:", error);
  });

function initMap() {
    var mapCenter = {lat: 37.48420796254918, lng:-79.01050225290268};
    map = new google.maps.Map(document.getElementById('map'), {
        zoom: 8,
        center: mapCenter,
    });

    fetch('stop_n_grab_list.json')
        .then(response => response.json())
        .then(data => {
            data.forEach(business => {
                var marker = new google.maps.Marker({
                    position: {lat: parseFloat(business.lat), lng: parseFloat(business.lng)},
                    map: map,
                    title: business.name
                });

                if(business.state == "Pennsylvania"){
                    PAmarkers.push(marker);
                }
                else{
                    markers.push(marker);
                }
           
             });
        });

    map.addListener('click', function() {
        hideNotPA();
    });

}

var show_hide = 0;

function hideNotPA() {
    if(show_hide == 0){
        for (var i = 0; i < markers.length; i++) {
            markers[i].setMap(null);
        }
        show_hide = 1;
    }
    else if(show_hide == 1){
        for (var i = 12; i < PAmarkers.length; i++) {
            PAmarkers[i].setMap(null);
        }
        show_hide = 2;
    }
    else{
        for (var i = 0; i < markers.length; i++) {
            markers[i].setMap(map);
        }
        for (var i = 12; i < PAmarkers.length; i++) {
            PAmarkers[i].setMap(map);
        }
        show_hide = 0;
    }

}