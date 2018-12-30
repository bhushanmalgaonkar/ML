const fs = require('fs');
const request = require('request');

var apikey = JSON.parse(fs.readFileSync('apikey', 'utf8'));

var geocodeAddress = (address, callback) => {
    var encodedAddress = encodeURIComponent(address);
    request({
        uri: `https://maps.googleapis.com/maps/api/geocode/json?address=${encodedAddress}&key=${apikey['maps']}`,
        json: true
    }, (error, response, body) => {
        if (error) {
            callback('Unable to connect', undefined);
        } else if (body.status === 'ZERO_RESULTS') {
            callback('Unable to find the address', undefined);
        } else if (body.status !== 'OK') {
            callback(body.error_message, undefined);
	} else if (body.status === 'OK') {
            callback(undefined, {
                address: body.results[0].formatted_address,
                lat: body.results[0].geometry.location.lat,
                lng: body.results[0].geometry.location.lng
            });
        }
    });
};


module.exports = {
    geocodeAddress
};
