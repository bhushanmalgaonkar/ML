var somePromise = new Promise((resolve, reject) => {
    resolve('It worked!');
    // reject(':(');
});

somePromise.then((message) => {
    console.log('Fulfilled: ' + message);
}, (errorMessage) => {
    console.log('Failed: ' + errorMessage);
});



// to allow input/outputs -> create a function returning promise
var asyncAdd = (a, b) => {
    return new Promise((resolve, reject) => {
        if (typeof a === 'number' && typeof b === 'number') {
            resolve(a + b);
        } else {
            reject('Both a and b must be numbers');
        }
    });
};

asyncAdd(2, 3).then((addition) => {
    console.log('Addition: ' + addition);
}, (errorMessage) => {
    console.log(errorMessage);
});


// promise chaining
asyncAdd(2, 3).then((addition) => {
    console.log('1st addition: ' + addition);
    return asyncAdd(addition, 5);
}).then((addition) => {
    console.log('2nd addition: ' + addition);
}).catch((errorMessage) => {
    console.log(errorMessage);
});