var map;
var cam_markers = [];
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
    var mapCenter = {lat: 39.991334775741244, lng:-75.20318736382991};
    // mapCenter = {lat: 39.145255217338594, lng: -99.98531305929917};
    map = new google.maps.Map(document.getElementById('map'), {
        zoom: 10,
        // zoom: 4,
        center: mapCenter,
    });

    fetch('camera_list.json')
        .then(response => response.json())
        .then(data => {
            data.forEach(camera => {
                var marker = new google.maps.Marker({
                    position: {lat: parseFloat(camera.lat), lng: parseFloat(camera.lng)},
                    map: map,
                });
                    // Add circle
                var circle = new google.maps.Circle({
                    center: {lat: parseFloat(camera.lat), lng: parseFloat(camera.lng)},
                    map: map,
                    radius: parseFloat(camera.radius)*5,
                    // strokeWeight: 0.5,
                    // fillOpacity: 0.6,
                    // fillColor: '#AA0000',
                    strokeWeight: 0.0,
                    fillOpacity: 1.0,
                    // fillColor: '#f6ff00'
                    fillColor: 'red',
                    radius: 10.0

                });
                // console.log(camera.lat,camera.lng)
                // circle.bindTo('center', marker, 'position');

                // cam_markers.push(marker);           
             });
        });

    // var circle = new google.maps.Circle({
    //     map: map,
    //     center: {lat: 39.991334775741244, lng: -75.20318736382991},
    //     fillColor: 'blue',
    //     fillOpacity: 0.6,
    //     strokeWeight: 0.5,
    // });

    // // Initial radius for the circle at the default zoom level
    // var baseRadius = 5000;  // e.g., 500 km
    // circle.setRadius(baseRadius);

    // // Update circle's radius when the zoom level changes
    // google.maps.event.addListener(map, 'zoom_changed', function() {
    //     var zoomLevel = map.getZoom();
    //     var newRadius = baseRadius * Math.pow(2, (10 - zoomLevel));  // Adjust as needed
    //     circle.setRadius(newRadius);
    // });

    // map.addListener('click', function() {
    //     hideNotPA();
    // });

}

// var show_hide = 0;

// function hideNotPA() {
//     if(show_hide == 0){
//         for (var i = 0; i < markers.length; i++) {
//             markers[i].setMap(null);
//         }
//         show_hide = 1;
//     }
//     else if(show_hide == 1){
//         for (var i = 12; i < PAmarkers.length; i++) {
//             PAmarkers[i].setMap(null);
//         }
//         show_hide = 2;
//     }
//     else{
//         for (var i = 0; i < markers.length; i++) {
//             markers[i].setMap(map);
//         }
//         for (var i = 12; i < PAmarkers.length; i++) {
//             PAmarkers[i].setMap(map);
//         }
//         show_hide = 0;
//     }

// }