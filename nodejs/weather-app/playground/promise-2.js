const fs = require('fs');
const request = require('request');



var geocodeAddress = (address) => {
    return new Promise((resolve, reject) => {
        var apikey = JSON.parse(fs.readFileSync('apikey', 'utf8'));
        var encodedAddress = encodeURIComponent(address);
        request({
            uri: `https://maps.googleapis.com/maps/api/geocode/json?address=${encodedAddress}&key=${apikey['maps']}`,
            json: true
        }, (error, response, body) => {
            if (error) {
                reject('Unable to connect');
            } else if (body.status === 'ZERO_RESULTS') {
                reject('Unable to find the address');
            } else if (body.status === 'OK') {
                resolve({
                    address: body.results[0].formatted_address,
                    lat: body.results[0].geometry.location.lat,
                    lng: body.results[0].geometry.location.lng
                });
            }
        });
    });
};

geocodeAddress('leaning tower of pisa').then((result) => {
    console.log(JSON.stringify(result));
}).catch((errorMessage) => {
    console.log(errorMessage);
});
