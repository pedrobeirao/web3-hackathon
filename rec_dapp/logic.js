// connect to Moralis server

const serverUrl = "https://31vwz9obhvgi.usemoralis.com:2053/server";
const appId = "4fbXErJ8gKuvkE2TS6OJlIVOooxNFtZ5XNL5MqCY";
Moralis.start({ serverUrl, appId });

async function login() {
    let user = Moralis.User.current();
    if (!user) {
    user = await Moralis.authenticate();
    }
    console.log("logged in user:", user);
}

async function logOut() {
    await Moralis.User.logOut();
    console.log("logged out");
}

document.getElementById("btn-login").onclick = login;
document.getElementById("btn-logout").onclick = logOut;
